# PPO Text Ablation Report

Status: `completed`

## Summary

| strategy | total_return | sharpe | max_drawdown | source_file |
| --- | --- | --- | --- | --- |
| PPO without text | -0.0729 | -0.214 | -0.202 | ppo_without_text_BENCHMARK\benchmark_summary.csv |
| Equal-weight Buy&Hold | -0.0858 | -0.327 | -0.208 | ppo_without_text_BENCHMARK\benchmark_summary.csv |
| DJI baseline | -0.1074 | -0.423 | -0.219 | ppo_without_text_BENCHMARK\benchmark_summary.csv |
| PPO with text | -0.196028 | -0.915197 | -0.222799 | artifacts\ppo_with_text_run\results\ppo_with_text_summary_for_comparison.csv |

## Delta: PPO with text minus PPO without text

| metric | ppo_without_text | ppo_with_text | delta |
| --- | --- | --- | --- |
| total_return | -0.0729 | -0.196028 | -0.123128 |
| sharpe | -0.214 | -0.915197 | -0.701197 |
| max_drawdown | -0.202 | -0.222799 | -0.0207986 |
