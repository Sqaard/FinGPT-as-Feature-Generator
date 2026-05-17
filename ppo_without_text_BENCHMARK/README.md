# PPO Without Text Benchmark

This is the frozen baseline for the first text-feature ablation.

## Model

- `ppo_custom_custom_350000_steps.zip`

## Baseline metrics

| Strategy | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| PPO without text | -7.29% | -0.214 | -20.2% |
| Equal-weight Buy&Hold | -8.58% | -0.327 | -20.8% |
| DJI baseline | -10.74% | -0.423 | -21.9% |

Use these as the first rule-based comparison targets after training PPO with
text features.

