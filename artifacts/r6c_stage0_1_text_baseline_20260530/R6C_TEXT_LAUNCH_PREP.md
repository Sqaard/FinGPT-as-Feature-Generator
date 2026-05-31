# R6c Stage0.1 Text Launch Preparation

## Baseline Comparison

- Old PPO without text: total_return=-0.072900, sharpe=-0.214000, max_drawdown=-0.202000.
- New R6c frozen baseline: total_return=-0.019227, sharpe=-0.101488, max_drawdown=-0.117761.

Interpretation: this is a baseline handoff comparison, not a final statistical claim. The old PPO and R6c Stage0.1 differ in action semantics and execution logic, but the frozen 2022-2023 window is aligned enough for launch triage.

## Text Experiment

- Integration: DeepSeek v2 compact text features are appended as fixed numeric state columns.
- Normalization: Stage0.1 fold-train-only scaler will fit market and text columns together.
- Source RL repo: read-only. All runnable files are copied into `rl_stage0_1_r6c_project/`.

## Launch Commands

Smoke:

```powershell
cd "artifacts\r6c_stage0_1_text_baseline_20260530\rl_stage0_1_r6c_project"
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1 --folds fold_2021 --smoke-test --force
```

Full 4-fold screening:

```powershell
cd "artifacts\r6c_stage0_1_text_baseline_20260530\rl_stage0_1_r6c_project"
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

Validation summary extracted from new baseline zips: `C:\Users\ivanp\OneDrive\Рабочий стол\доки+черчи\ITMO\2_sem\FinRL_Tsinghua\FinGPT-as-Feature-Generator_repo\artifacts\r6c_stage0_1_text_baseline_20260530\baseline_r6c_validation\r6c_validation_results_all_folds.csv`.

## Smoke Validation

The local copied project was smoke-tested with the text config on `fold_2021`.

- Command: `python -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1 --folds fold_2021 --smoke-test --force`
- Status: passed.
- Smoke validation return: `0.05065587356876389`.
- Smoke validation Sharpe: `1.1430422832340847`.
- Smoke validation max drawdown: `-0.0297832803007837`.
- Smoke turnover L1 mean: `0.005622188071587286`.
- Smoke output root: `rl_stage0_1_r6c_project/artifacts/stage0_1_text/r6c_deepseek_v2_text_state_concat_smoke/`.

## Post-Training Result

Full 4-fold training completed in `rl_stage0_1_r6c_project/artifacts/stage0_1_text/r6c_deepseek_v2_text_state_concat/`.

Walk-forward validation against the R6c baseline:

- Baseline mean validation return: `0.100488`; text mean validation return: `0.091020`; delta: `-0.009468`.
- Baseline mean validation Sharpe: `1.093867`; text mean validation Sharpe: `1.061245`; delta: `-0.032622`.
- Baseline mean validation max drawdown: `-0.073387`; text mean validation max drawdown: `-0.071319`; delta: `+0.002068`.
- Baseline mean turnover L1: `0.008442`; text mean turnover L1: `0.008312`; delta: `-0.000130`.

Frozen OOS `fold_2021` result against the selected R6c frozen baseline:

- Baseline frozen return: `-0.019227`; text frozen return: `-0.016101`; delta: `+0.003126`.
- Baseline frozen Sharpe: `-0.101488`; text frozen Sharpe: `-0.064038`; delta: `+0.037450`.
- Baseline frozen max drawdown: `-0.117761`; text frozen max drawdown: `-0.120434`; delta: `-0.002673`.
- Baseline frozen turnover L1: `0.007443`; text frozen turnover L1: `0.007619`; delta: `+0.000177`.

Interpretation: the current DeepSeek v2 text10 state-concat features do not improve walk-forward validation, but they slightly improve frozen OOS return and Sharpe on the selected `fold_2021` model while making drawdown marginally worse. This is not strong enough to declare success; it is enough to justify one controlled 3-seed confirmation only if frozen OOS Sharpe is the primary triage criterion.
