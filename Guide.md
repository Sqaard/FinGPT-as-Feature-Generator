## First Feature Set: Text To Numerical

The LLM should fill these 10 features for every document.

1. `text_relevance_to_portfolio` in `[0, 1]`  
   How relevant this document is for the portfolio/ticker.

2. `sentiment_direction` in `[-1, 1]`  
   Negative, neutral, or positive financial tone.

3. `price_impact_direction` in `[-1, 1]`  
   Expected directional pressure from the document, not a trading decision.

4. `risk_intensity` in `[0, 1]`  
   How strongly the document increases or discusses downside risk.

5. `uncertainty_intensity` in `[0, 1]`  
   How uncertain, conditional, or unclear the outlook is.

6. `opportunity_intensity` in `[0, 1]`  
   Growth, margin upside, demand strength, buybacks, dividends, or other upside.

7. `forward_looking_intensity` in `[0, 1]`  
   Guidance, outlook, expectations, or future-looking language.

8. `earnings_guidance_impact` in `[-1, 1]`  
   Impact of earnings, revenue, EPS, margins, or guidance.

9. `macro_financial_conditions_impact` in `[-1, 1]`  
   Impact of rates, credit, inflation, volatility, labor, energy, or broad macro.

10. `company_event_risk_impact` in `[-1, 1]`  
    Impact of legal, regulatory, supply-chain, demand, margin, or company event
    risk.


## Folder Structure

```text
Toy_Example/
  data/
    train_2010_2021/
    test_2021_2023/
    processed_final_fixed_external_lagclean_full.csv
  scripts/
    01_extract_text_features_mistral.py
    02_merge_text_features_with_prices.py
    03_train_backtest_ppo_with_text.py
    04_compare_with_benchmark.py
  ppo_without_text_BENCHMARK/
    ppo_custom_custom_350000_steps.zip
    benchmark_summary.csv
    results/
    figures/
  rl_stage0_project/
    README.md
    stage0_audit/
      stage0_model_pipeline.py
      stage0_methodology.py
  artifacts/
    text_features_mistral.csv
    processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv
    ppo_with_text_run/
    ppo_text_vs_benchmark/
  requirements_runtime.txt
  train_ppo_with_text.ipynb
```

You  need a Python environment with FinRL, Stable-Baselines3, PyTorch, pandas, numpy, and pyfolio.
`requirements_runtime.txt` records the runtime versions used for the current
run.

## Step 1 - Extract Text Features

This is the Mistral extraction script:

```text
scripts/01_extract_text_features_mistral.py
```

Set your API key:

```powershell
$env:MISTRAL_API_KEY="your_key_here"
```
Full run:

```powershell
python .\scripts\01_extract_text_features_mistral.py `
  --input .\data\train_2010_2021 `
  --input .\data\test_2021_2023 `
  --output .\artifacts\text_features_mistral.csv
```

## Step 2 - Merge With The Main Price Panel

The main CSV has about 96k daily stock rows. We have fewer documents than daily
stock rows, so the script aggregates document features by date and ticker:

```text
document available_at -> first trading date >= available_at date
matched ticker docs -> that ticker
MARKET/macro docs -> all tickers on that date
missing text features -> 0
```

Run with real Mistral features:

```powershell
python .\scripts\02_merge_text_features_with_prices.py `
  --base-panel .\data\processed_final_fixed_external_lagclean_full.csv `
  --text-features .\artifacts\text_features_mistral.csv `
  --output .\artifacts\processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv `
  --manifest .\artifacts\merge_manifest_mistral.json
```

## Step 3 - Train And Backtest PPO With Text

This script reuses the local copied Stage0 PPO environment code from
`rl_stage0_project/`. It trains a new PPO model from the merged panel with the
10 text columns, then writes validation/test curves, summaries, actions, and a
compact comparison CSV.

Recommended command:

```powershell
python .\scripts\03_train_backtest_ppo_with_text.py `
  --panel .\artifacts\processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv `
  --output-dir .\artifacts\ppo_with_text_run `
  --timesteps 350000
```

Main outputs:

- `artifacts/ppo_with_text_run/checkpoints/ppo_with_text_custom_custom_350000_steps.zip`;
- `artifacts/ppo_with_text_run/results/ppo_with_text_test_curve.csv`;
- `artifacts/ppo_with_text_run/results/ppo_with_text_test_summary.csv`;
- `artifacts/ppo_with_text_run/results/ppo_with_text_summary_for_comparison.csv`.

## Step 4 - Compare With The Frozen Benchmark

Run:

```powershell
python .\scripts\04_compare_with_benchmark.py `
  --benchmark-csv .\ppo_without_text_BENCHMARK\benchmark_summary.csv `
  --ppo-with-text-csv .\artifacts\ppo_with_text_run\results\ppo_with_text_summary_for_comparison.csv `
  --output-dir .\artifacts\ppo_text_vs_benchmark
```

The comparison script creates:

- `artifacts/ppo_text_vs_benchmark/report.md`;
- `artifacts/ppo_text_vs_benchmark/results/comparison_summary.csv`;
- `artifacts/ppo_text_vs_benchmark/results/comparison_deltas.csv`;
- `artifacts/ppo_text_vs_benchmark/figures/return_sharpe_drawdown.svg`.

## Baseline To Beat

Current PPO without text:

```text
RL custom_custom:        return -7.29%,  Sharpe -0.214, max DD -20.2%
Equal-weight Buy&Hold:  return -8.58%,  Sharpe -0.327, max DD -20.8%
DJI baseline:           return -10.74%, Sharpe -0.423, max DD -21.9%
```

## Current Run Status

The first full Mistral extraction has already been completed:

- `artifacts/text_features_mistral.csv`: 20,005 document-level rows with the 10
  text-to-numerical features.
- `artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv`:
  96,019 daily panel rows after merging text features into the PPO input panel.
- `artifacts/ppo_with_text_run/checkpoints/ppo_with_text_custom_custom_350000_steps.zip`:
  the first trained PPO-with-text model.

The first ablation result is negative:

```text
PPO without text: return -7.29%,  Sharpe -0.214, max DD -20.2%
PPO with text:    return -19.60%, Sharpe -0.915, max DD -22.28%
```

This does not prove that text is useless. It means that the first raw set of 10
Mistral features, added directly into the PPO state, hurts this policy. The next
research step is to normalize and compress the text features, then run feature
ablation to find which features or feature groups are harmful and which ones may
carry useful signal.

## Step 5 - DRL Lane: Train-Only Text Normalization

GitHub issue: `#5`.

```powershell
python .\scripts\05_normalize_text_features_train_only.py
```

Outputs:

- `artifacts/normalized_text_panels/train_only_scaler_stats.json`;
- one normalized panel per method: `zscore`, `robust`, `clipped`,
  `zscore_clipped`, `robust_clipped`;
- `reports/drl_train_only_normalization.md`.

All scaler parameters are fitted on `2010-01-04` to `2021-09-30` only.
Validation/test rows are transformed with the frozen train-window parameters.

## Step 6 - DRL Lane: PPO-Side Text Integration

GitHub issue: `#6`.

```powershell
python .\scripts\06_build_ppo_text_integration_configs.py
```

Outputs:

- `artifacts/ppo_text_integration_configs/experiment_matrix.csv`;
- `artifacts/ppo_text_integration_configs/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_text_risk_overlay.csv`;
- `reports/drl_ppo_text_integration.md`.

The train wrapper accepts these strategies:

```powershell
python .\scripts\03_train_backtest_ppo_with_text.py `
  --panel .\artifacts\normalized_text_panels\processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_robust_clipped.csv `
  --text-integration-strategy two_branch_policy `
  --timesteps 350000
```

Available strategies:

- `state_concat`: text is appended to the PPO state;
- `market_only`: control run, text columns excluded from the state;
- `text_risk_overlay`: text changes risk control through `turbulence_text_overlay`;
- `two_branch_policy`: market and text features pass through separate SB3 MLP branches.

## Step 7 - DRL Lane: Document And Source Quality

GitHub issue: `#7`.

```powershell
python .\scripts\07_score_document_source_quality.py
```

Outputs:

- `artifacts/document_source_quality/document_quality.csv`;
- `artifacts/document_source_quality/source_family_quality.csv`;
- `reports/drl_document_source_quality.md`.

The score combines relevance, signal density, timestamp integrity, source
reliability, and extraction status. A next-day return diagnostic is included for
ticker-level documents, but it is only an event-style sanity check, not an alpha
claim.

## Step 8 - DRL Lane: CHRL Text Feature Group Ablation

Imported from the `Features` branch and adapted to the current CHRL naming.

```powershell
python .\scripts\13_run_chrl_text_feature_group_ablation.py
```

Key outputs:

- `artifacts/chrl_text_feature_group_ablation/final_report.md`;
- `artifacts/chrl_text_feature_group_ablation/results/chrl_ablation_summary.csv`;
- `reports/chrl_text_feature_group_ablation.md`.

Best current screening result on `fold_2021` frozen OOS:

```text
CHRL model baseline:                    return -1.92%, Sharpe -0.1015, max DD -11.78%
CHRL model + all raw text10:             return -1.61%, Sharpe -0.0640, max DD -12.04%
CHRL + forward/earnings compact group:   return -0.65%, Sharpe  0.0105, max DD -12.37%
```

The compact forward/earnings group contains:

```text
text_earnings_pressure
text_numeric_evidence_density
text_signal_confidence
```

This result is a feature-selection signal, not final evidence. It should be
re-run on multiple seeds/folds before being treated as a production PPO input.

## Step 9 - LLM Lane Methodology Port: R7g Ideas

Imported from the `LLM` branch as:

- `reports/r7g_regime_aware_methodology.md`.

This is a methodology reference, not runnable code in this branch yet. The
portable ideas to implement next are:

- split text into company, macro, sector, and market-breadth channels;
- avoid broadcasting macro text into every stock feature without a separate
  channel;
- use a two-tower market/text encoder for CHRL instead of plain concatenation;
- test macro text as root risk/cash control rather than stock-level alpha;
- add regime-aware reward shaping where stress increases drawdown penalty and
  reduces return-chasing pressure.
