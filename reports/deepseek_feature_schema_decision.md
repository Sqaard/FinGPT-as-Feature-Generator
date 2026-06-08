# DeepSeek Feature Schema Decision

The live DeepSeek comparison completed successfully: 192 / 192 schema-document extractions were valid.

## Decision

Use `feature_schema_deepseek_v2_ppo_compact.json` as the next full-corpus extraction contract.

It combines:

- the stable `ppo_state_compact_7` state features;
- three source-quality gates from `source_quality_8`.

## Why

| Candidate | Decision | Reason |
|---|---|---|
| `v1_baseline_10` | Keep as baseline | Best diagnostic score, but already produced a negative first PPO ablation and mixes broad concepts. |
| `ppo_state_compact_7` | Use as core | Compact, non-constant, and balanced across earnings releases, SEC exhibits, macro, and company IR. |
| `source_quality_8` | Use as gates | High signal density, but it mostly measures document quality, not alpha. |
| `literature_top_12` | Keep for research | Literature-backed, but larger than needed for first PPO rerun. |
| `risk_control_8` | Do not use directly | Too sparse; better as later overlay diagnostics. |
| `investor_operator_12` | Do not use directly | Too sparse on company IR and macro documents. |

## Chosen v2 features

| Feature | Role |
|---|---|
| `text_alpha_direction` | Directional text signal |
| `text_downside_risk` | Risk control |
| `text_uncertainty` | Confidence/risk control |
| `text_macro_stress` | Portfolio-wide stress |
| `text_earnings_pressure` | Fundamentals/guidance pressure |
| `text_balance_sheet_stress` | Leverage and liquidity risk |
| `text_signal_confidence` | LLM confidence in signal |
| `text_evidence_specificity` | Evidence gate |
| `text_numeric_evidence_density` | Numeric evidence gate |
| `text_boilerplate_intensity` | Noise gate |

## Next command

Run full-corpus extraction:

```powershell
python scripts/09_extract_deepseek_v2_features.py --restart
```

Then merge into the price panel:

```powershell
python scripts/02_merge_text_features_with_prices.py --text-features artifacts/text_features_deepseek_v2.csv --feature-schema feature_schema_deepseek_v2_ppo_compact.json --output artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2.csv --manifest artifacts/merge_manifest_deepseek_v2.json
```

Then create train-only normalized variants:

```powershell
python scripts/05_normalize_text_features_train_only.py --panel artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2.csv --text-feature-schema feature_schema_deepseek_v2_ppo_compact.json --output-dir artifacts/normalized_text_panels_deepseek_v2 --report reports/drl_train_only_normalization_deepseek_v2.md
```

Then validate the PPO handoff before training:

```powershell
python scripts/03_train_backtest_ppo_with_text.py --panel artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2.csv --text-feature-schema feature_schema_deepseek_v2_ppo_compact.json --dry-run --output-dir artifacts/ppo_with_text_deepseek_v2_dry_run
```

This is still an extraction-stage decision, not a PPO performance claim.
