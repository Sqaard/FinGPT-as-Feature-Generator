# DRL Issue #5: Train-only text normalization

Goal: make text features safer for PPO without leaking validation or test information.

Source panel: `artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2.csv`
Scaler statistics: `artifacts/normalized_text_panels_deepseek_v2/train_only_scaler_stats.json`

## Leakage rule

All means, standard deviations, medians, IQRs, and clipping bounds are fitted on the train window only.
The same fitted parameters are then applied to every later row.

## Generated panels

| method | output | train_rows | validation_test_rows | train_max_abs_value | validation_test_max_abs_value |
| --- | --- | --- | --- | --- | --- |
| zscore | artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_zscore.csv | 85753 | 10266 | 50.2964 | 33.9962 |
| robust | artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_robust.csv | 85753 | 10266 | 9.5 | 9.80555 |
| clipped | artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_clipped.csv | 85753 | 10266 | 0.76 | 0.588889 |
| zscore_clipped | artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_zscore_clipped.csv | 85753 | 10266 | 3 | 3 |
| robust_clipped | artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_robust_clipped.csv | 85753 | 10266 | 3 | 3 |

## Interpretation

- `zscore`: useful when PPO benefits from centered Gaussian-like features.
- `robust`: safer when filings create heavy-tailed one-day spikes.
- `clipped`: tests whether raw features were mostly fine but outliers hurt PPO.
- clipped scaled variants are the first candidates for a stable DRL run.