# Cross-Model Text Effect Comparison

This table compares text impact within each PPO architecture family. It should not be read as a clean head-to-head architecture benchmark, because `custom_custom` and `R6c` have different action/execution mechanics.

## Model Rows

| model | period | return | Sharpe | max DD | turnover | source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| custom_custom | test | -7.29% | -0.2140 | -20.20% |  | `ppo_without_text_BENCHMARK\benchmark_summary.csv` |
| custom_custom&text10 | test | -19.60% | -0.9152 | -22.28% |  | `artifacts\ppo_with_text_run\results\ppo_with_text_test_summary.csv` |
| R6c | frozen_oos_fold_2021 | -1.92% | -0.1015 | -11.78% | 0.0074 | `C:\Users\ivanp\RL for Time-Series Forecasting\data_RLagent_for_Joseph\artifacts\stage4\R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph\frozen_test_behavior_log_daily.csv` |
| R6c&text10 | frozen_oos_fold_2021 | -1.61% | -0.0640 | -12.04% | 0.0076 | `artifacts\r6c_stage0_1_text_baseline_20260530\rl_stage0_1_r6c_project\artifacts\stage0_1_text\r6c_deepseek_v2_text_frozen_oos\frozen_oos_results.csv` |

## Text Deltas

| family | return delta | Sharpe delta | max DD delta | direction |
| --- | ---: | ---: | ---: | --- |
| custom_custom | -12.31% | -0.7012 | -2.08% | worse |
| R6c | 0.31% | 0.0374 | -0.27% | better |

## Readout

- On `custom_custom`, text10 made the test result materially worse: return and Sharpe both fell strongly.
- On `R6c`, text10 slightly improved return and Sharpe, while max drawdown became slightly worse.
- Current evidence says text features interact better with the R6c hierarchy than with the old flat `custom_custom` policy, but the R6c gain is still small and should be treated as screening evidence, not final proof.
