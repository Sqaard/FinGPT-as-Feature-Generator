# R6c Text Features: Post-Training Next Steps

## Current Result

The R6c text model trained successfully with DeepSeek v2 compact text10 features.

Validation comparison is mixed:

- Mean validation return: baseline `0.100488`, text `0.091020`, delta `-0.009468`.
- Mean validation Sharpe: baseline `1.093867`, text `1.061245`, delta `-0.032622`.
- Mean validation max drawdown: baseline `-0.073387`, text `-0.071319`, delta `+0.002068`.
- Mean validation turnover L1: baseline `0.008442`, text `0.008312`, delta `-0.000130`.

Frozen OOS `fold_2021` comparison is slightly positive on return and Sharpe:

- Frozen return: baseline `-0.019227`, text `-0.016101`, delta `+0.003126`.
- Frozen Sharpe: baseline `-0.101488`, text `-0.064038`, delta `+0.037450`.
- Frozen max drawdown: baseline `-0.117761`, text `-0.120434`, delta `-0.002673`.
- Frozen turnover L1: baseline `0.007443`, text `0.007619`, delta `+0.000177`.

## Decision

Do not declare the text feature set successful yet. It does not improve the main 4-fold validation metrics, and the frozen OOS improvement is small.

The next controlled step is one of:

1. If frozen OOS Sharpe is the triage criterion, run the same experiment on 3 seeds.
2. If validation Sharpe is the triage criterion, stop this feature set and build a leaner/action-aware text feature set before spending more compute.

## Recommended Path

Use this model as a weak positive OOS signal, not as a finished ablation.

Next experiments should compare:

- `R6c_base`
- `R6c_plus_text10_state_concat`
- `R6c_plus_text_action_primitive_v1`
- `R6c_plus_text_risk_gate_v1`

The highest-priority improvement is not adding more raw text columns. It is moving text into regime/risk/action-aligned features or gates:

- downside/risk stress interaction with existing `risk_stress`;
- recovery/rerisk support interaction with existing `recovery_score`;
- text confidence/evidence gates that suppress weak or boilerplate filings;
- action-primitive features targeted at derisk/rerisk/non-flat regimes.

## Generated Files

- `rl_stage0_1_r6c_project/artifacts/stage0_1_text/r6c_deepseek_v2_text_state_concat/walk_forward_validation_results.csv`
- `rl_stage0_1_r6c_project/artifacts/stage0_1_text/r6c_deepseek_v2_text_state_concat/walk_forward_variant_summary.csv`
- `rl_stage0_1_r6c_project/artifacts/stage0_1_text/r6c_deepseek_v2_text_frozen_oos/frozen_oos_results.csv`
- `r6c_text_frozen_vs_baseline_fold2021.csv`
