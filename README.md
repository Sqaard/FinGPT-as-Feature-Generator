# FinGPT as Feature Generator

Small, portable experiment package for testing whether causal financial text
features improve a Dow 30 PPO trading agent.

## Snapshot

| Block | Current artifact |
|---|---|
| Documents | `20,005` timestamped text rows |
| PPO panel | `96,019` daily stock rows |
| Text extractor | `mistral-small-latest` |
| Text features | `10` numerical PPO features |
| Baseline | PPO without text, `350k` steps |
| First result | negative ablation; use it to debug features |
| Large files | tracked with Git LFS |

## Pipeline

```mermaid
flowchart LR
  A["SEC / Macro / Company IR documents"] --> B["LLM text-to-numerical extraction"]
  B --> C["Daily ticker panel merge"]
  C --> D["PPO with text features"]
  E["PPO without text benchmark"] --> F["Ablation report"]
  D --> F
  F --> G["Feature normalization / ablation / PPO-side changes"]
```

## First Ablation

| Strategy | Return | Sharpe | Max DD |
|---|---:|---:|---:|
| PPO without text | `-7.29%` | `-0.214` | `-20.2%` |
| PPO with Mistral text | `-19.60%` | `-0.915` | `-22.28%` |

![PPO text ablation](artifacts/ppo_text_vs_benchmark/figures/return_sharpe_drawdown.svg)

## DRL R6c Frozen OOS

| setup | OOS return | OOS Sharpe | OOS max DD |
|---|---:|---:|---:|
| `custom_custom` | `-7.29%` | `-0.2140` | `-20.20%` |
| `custom_custom+text10` | `-19.60%` | `-0.9152` | `-22.28%` |
| `R6c baseline` | `-1.92%` | `-0.1015` | `-11.78%` |
| `R6c + raw_text10` | `-1.61%` | `-0.0640` | `-12.04%` |

The current best text candidate for the DRL branch is `R6c + raw_text10`.
It improves frozen OOS return and Sharpe versus the R6c baseline, while max
drawdown is slightly worse.

## Main Files

| Path | Purpose |
|---|---|
| `Guide.md` | full step-by-step run guide |
| `feature_schema.json` | first 10 text-to-numerical features |
| `data/train_2010_2021/` | train text documents |
| `data/test_2021_2023/` | test text documents |
| `artifacts/text_features_mistral.csv` | completed Mistral extraction |
| `artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv` | merged PPO panel |
| `ppo_without_text_BENCHMARK/` | frozen baseline model and metrics |
| `rl_stage0_project/` | copied local PPO Stage0 code |
| `scripts/05_normalize_text_features_train_only.py` | train-only text scalers |
| `scripts/06_build_ppo_text_integration_configs.py` | PPO-side text experiment matrix |
| `scripts/07_score_document_source_quality.py` | document/source quality scoring |
| `train_ppo_with_text.ipynb` | notebook wrapper for training and comparison |

## Commands

| Step | Command |
|---|---|
| Extract | `python scripts/01_extract_text_features_mistral.py --input data/train_2010_2021 --input data/test_2021_2023 --output artifacts/text_features_mistral.csv` |
| Merge | `python scripts/02_merge_text_features_with_prices.py --base-panel data/processed_final_fixed_external_lagclean_full.csv --text-features artifacts/text_features_mistral.csv --output artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv` |
| Train | `python scripts/03_train_backtest_ppo_with_text.py --timesteps 350000` |
| Compare | `python scripts/04_compare_with_benchmark.py` |
| Normalize text | `python scripts/05_normalize_text_features_train_only.py` |
| Build DRL variants | `python scripts/06_build_ppo_text_integration_configs.py` |
| Score sources | `python scripts/07_score_document_source_quality.py` |

## Team Lanes

| Branch | Owner lane | Focus |
|---|---|---|
| `LLM` | 蔡志成 | extraction quality, prompts, model/API comparison |
| `Features` | Tianyi Tan | feature selection, smoothing, macro vs company split |
| `DRL` | Vanya | PPO-side changes, source quality, evaluation loop |

## Next Experiments

| Experiment | Why |
|---|---|
| Train-only normalization | PPO is scale-sensitive |
| Feature-group ablations | find harmful vs useful feature families |
| Macro vs company features | different text sources should not be mixed blindly |
| 3d / 5d / 21d decay | filings and earnings releases matter beyond one day |
| Reward/risk text use | text may work better as risk control than raw state |
| Two-branch PPO policy | market features and text features need separate encoders |

## DRL Branch Issue Outputs

| GitHub issue | Implemented output |
|---|---|
| `#5` train-only normalization | `artifacts/normalized_text_panels/` and `reports/drl_train_only_normalization.md` |
| `#6` PPO-side text integration | `artifacts/ppo_text_integration_configs/experiment_matrix.csv` and `reports/drl_ppo_text_integration.md` |
| `#7` document/source quality | `artifacts/document_source_quality/` and `reports/drl_document_source_quality.md` |

The train wrapper now supports `--text-integration-strategy` with
`state_concat`, `market_only`, `text_risk_overlay`, and `two_branch_policy`.
