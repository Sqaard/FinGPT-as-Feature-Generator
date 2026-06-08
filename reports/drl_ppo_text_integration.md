# DRL Issue #6: PPO-side text integration strategies

Goal: test whether text works better outside raw feature concatenation.

## Experiment matrix

| variant | strategy | panel | purpose |
| --- | --- | --- | --- |
| state_concat_raw | state_concat | artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv | current baseline: raw text directly in PPO state |
| state_concat_robust_clipped | state_concat | artifacts/normalized_text_panels/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_robust_clipped.csv | same architecture, train-only robust clipped text |
| market_only_control | market_only | artifacts/normalized_text_panels/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_robust_clipped.csv | control run: panel contains text but PPO state excludes text columns |
| text_risk_overlay | text_risk_overlay | artifacts/ppo_text_integration_configs/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_text_risk_overlay.csv | text affects risk control through turbulence overlay |
| two_branch_policy | two_branch_policy | artifacts/normalized_text_panels/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_robust_clipped.csv | market/text separate encoders before PPO policy/value heads |

## Risk overlay

The risk-overlay variant converts text risk, uncertainty, macro stress, and event risk into a `turbulence_text_overlay` column.
This lets the existing FinRL risk-control path react to text without changing the environment internals.

```json
{
  "output": "artifacts/ppo_text_integration_configs/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_text_risk_overlay.csv",
  "train_turbulence_p95": 97.96642405420558,
  "risk_overlay_weight": 0.5,
  "text_risk_score_mean": 0.2816665958935525,
  "text_risk_score_p95": 1.2742424242424235,
  "overlay_turbulence_p95": 133.7131747903412
}
```

## Two-branch policy prototype

`scripts/03_train_backtest_ppo_with_text.py --text-integration-strategy two_branch_policy` now builds a custom SB3 feature extractor.
It sends market features and text features through separate MLP branches before PPO receives the merged representation.