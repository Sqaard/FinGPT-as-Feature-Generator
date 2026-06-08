# DRL Issue #5: Train-only text normalization

Goal: make text features safer for PPO without leaking validation or test information.

Source panel: `artifacts/merged_deepseek_v2_dryrun_smoke.csv`
Scaler statistics: `artifacts/normalized_text_panels_deepseek_v2_smoke/train_only_scaler_stats.json`

## Leakage rule

All means, standard deviations, medians, IQRs, and clipping bounds are fitted on the train window only.
The same fitted parameters are then applied to every later row.

## Generated panels

| method | output | train_rows | validation_test_rows | train_max_abs_value | validation_test_max_abs_value |
| --- | --- | --- | --- | --- | --- |
| robust_clipped | artifacts/normalized_text_panels_deepseek_v2_smoke/merged_deepseek_v2_dryrun_smoke_robust_clipped.csv | 85753 | 10266 | 0 | 1 |

## Interpretation

- `zscore`: useful when PPO benefits from centered Gaussian-like features.
- `robust`: safer when filings create heavy-tailed one-day spikes.
- `clipped`: tests whether raw features were mostly fine but outliers hurt PPO.
- clipped scaled variants are the first candidates for a stable DRL run.