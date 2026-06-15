# CHRL Text Feature Group Ablation Report

## Data readiness

Prepared: `True`.
Raw CHRL/R6c panel: `artifacts/r6c_stage0_1_text_baseline_20260530/rl_stage0_1_r6c_project/artifacts/stage0_1/features/stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv`.
Missing text features: `[]`.

## Frozen OOS ablation table

| Strategy | Fold | Return | Sharpe | Max drawdown | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| CHRL model | fold_2021 | -1.92% | -0.1015 | -11.78% | baseline |
| CHRL model + all raw text10 | fold_2021 | -1.61% | -0.0640 | -12.04% | useful |
| Only risk / uncertainty | fold_2021 | -1.43% | -0.0539 | -12.15% | useful |
| Only sentiment / price impact | fold_2021 | -1.81% | -0.0588 | -13.57% | useful |
| Only forward-looking / earnings guidance | fold_2021 | -0.65% | 0.0105 | -12.37% | useful |
| Only macro financial conditions | fold_2021 | -1.62% | -0.0741 | -11.59% | useful |
| Top-3 train-correlation features | fold_2021 | -1.37% | -0.0349 | -12.92% | useful |

## Feature groups

- `risk_uncertainty`: Only risk / uncertainty = `text_downside_risk, text_uncertainty, text_balance_sheet_stress, text_boilerplate_intensity`
- `sentiment_price`: Only sentiment / price impact = `text_alpha_direction, text_signal_confidence, text_evidence_specificity`
- `forward_earnings`: Only forward-looking / earnings guidance = `text_earnings_pressure, text_numeric_evidence_density, text_signal_confidence`
- `macro_conditions`: Only macro financial conditions = `text_macro_stress, text_uncertainty`
- `topk_train_correlation`: Top-3 train-correlation features = `text_numeric_evidence_density, text_boilerplate_intensity, text_evidence_specificity`

## Top-k selection method

Top-k uses absolute Pearson correlation between each DeepSeek text feature and next-day same-ticker return on the selected fold's train window.

| feature | train_next_day_return_corr | abs_train_next_day_return_corr |
| --- | ---: | ---: |
| text_numeric_evidence_density | 0.02674 | 0.02674 |
| text_boilerplate_intensity | -0.0237193 | 0.0237193 |
| text_evidence_specificity | 0.0174184 | 0.0174184 |
| text_balance_sheet_stress | -0.0094851 | 0.0094851 |
| text_signal_confidence | 0.00734115 | 0.00734115 |
| text_alpha_direction | -0.00460893 | 0.00460893 |
| text_earnings_pressure | 0.00328129 | 0.00328129 |
| text_macro_stress | 0.00194214 | 0.00194214 |
| text_downside_risk | 0.00162114 | 0.00162114 |
| text_uncertainty | 0.00124294 | 0.00124294 |

## Interpretation

Useful groups: CHRL model + all raw text10, Only risk / uncertainty, Only sentiment / price impact, Only forward-looking / earnings guidance, Only macro financial conditions, Top-3 train-correlation features.
Noisy or mixed groups: none.
Harmful groups: none.

This CHRL rerun uses the newer DeepSeek compact text schema, so group names match task.md semantically but not one-to-one with the old Mistral feature columns.
