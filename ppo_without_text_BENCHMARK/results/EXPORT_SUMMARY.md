# Stage 0 Teacher Export Summary

This package is generated for Joseph's hidden-state and primitive-discovery work.

## Model

- model_id: `interpretable_no_gru_custom_custom_ppo_custom_custom_350000_steps`
- model_path: `ppo_without_text_BENCHMARK/ppo_custom_custom_350000_steps.zip`
- selected_config: `custom_custom`
- feature_set: `interpretable_no_gru`
- hook_mode: `auto_64_after_activation`
- hook_module_index: `5`
- hidden_dim: `64`

## Periods

| period | start | end | rows | obs_dim | hidden_dim | action_dim | missing_context |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 2010-01-04 | 2021-09-30 | 2957 | 610 | 64 | 29 | 0 |
| validation | 2021-10-01 | 2021-12-31 | 64 | 610 | 64 | 29 | 0 |
| test | 2022-01-03 | 2023-02-28 | 290 | 610 | 64 | 29 | 0 |
| cross_2010_2015 | 2010-01-04 | 2015-12-31 | 1510 | 610 | 64 | 29 | 0 |
| cross_2016_2021_train | 2016-01-04 | 2021-09-30 | 1447 | 610 | 64 | 29 | 0 |

## Notes

- `*_hidden` is the selected policy-branch hidden activation.
- `*_policy_latent_final` is also exported for layer-contract checks.
- `*_value_latent` is diagnostic only and should not be used as Joseph's primary behavior representation.
- `*_actions` are deterministic SB3 policy actions clipped to the environment action space.
- `*_executed_trade_shares` are the integer trade-share actions executed by the FinRL environment.
- `*_return_1d` is realized portfolio return from date t to t+1; the last row of each period is NaN.
- True PPO training-time `approx_kl` and `clip_fraction` are not reconstructable post hoc from a frozen model; collect them during future instrumented training if needed.
