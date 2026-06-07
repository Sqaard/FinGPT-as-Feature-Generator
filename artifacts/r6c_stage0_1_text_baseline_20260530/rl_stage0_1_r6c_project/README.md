# Stage 0 and Proposal-Aligned Interpretability Research

This is the active workspace for the renewed research plan:

> Self-supervised learning for reinforcement-learning agent interpretability.

The current direction is intentionally clean of the earlier G0/G1/G2 latent-action experiments. Those older folders remain useful as implementation references, but they are not the active methodology.

## Active Plan

- `PROPOSAL_ALIGNED_INTERPRETABILITY_RESEARCH_PLAN.md` is the current research plan aligned with `Project Proposal.pdf`.
- `METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md` is the canonical detailed methodology and implementation contract for the renewed project.
- `PROJECT_STRUCTURE.md` defines the active `configs/`, `src/`, `artifacts/`, `notebooks/`, and `reports/` layout.
- `reports/STAGE0_1_STABILIZED_PPO_RUN_GUIDE.md` describes the new stabilized weight-based PPO teacher run.
- `reports/STAGE0_1_RESCUE_BATCH_A_RUN_GUIDE.md` describes the first hierarchy-rescue experiment batch and Huawei package workflow.
- `reports/STAGE0_1_ROOT_SPLIT_EXPERIMENTS_2_3_PLAN.md` fixes the implementation plan for the root-split cash-vs-invested and separate-encoder experiments.
- `reports/STAGE0_1_ROOT_SPLIT_RUN_GUIDE.md` describes the Huawei package workflow for root-split experiments E2/E3.
- `reports/STAGE0_1_ROOT_SPLIT_EXECUTION_RUN_GUIDE.md` describes the Huawei package workflow for root-split execution experiments E4/E5.
- `reports/STAGE0_1_EXPERIMENTS_6_7_IMPLEMENTATION_PLAN.md` fixes the plan for learned Kp gates and sector Dirichlet-tree experiments.
- `reports/STAGE0_1_EXPERIMENTS_6_7_RUN_GUIDE.md` describes the Huawei package workflow for experiments E6/E7.
- `reports/STAGE0_1_EXPERIMENTS_8_9_IMPLEMENTATION_PLAN.md` fixes the plan for logistic-normal group distribution and discovered residual-correlation hierarchy experiments.
- `reports/STAGE0_1_EXPERIMENTS_8_9_RUN_GUIDE.md` describes the Huawei package workflow for experiments E8/E9.
- `reports/STAGE0_1_EXPERIMENTS_10_12_IMPLEMENTATION_PLAN.md` fixes the plan for bottom-up veto, projection safety, and interpretable style meta-policy experiments.
- `reports/STAGE0_1_EXPERIMENTS_10_12_RUN_GUIDE.md` describes the Huawei package workflow for experiments E10/E11/E12.
- `reports/STAGE1_IMPLEMENTATION_PLAN.md` defines the first implementation plan for Stage 1 primitive discovery.
- `reports/deepsearch/` stores canonical local copies of the DeepSearch research memos used to update Stage 0.1 methodology:
  - `deep-research-report-12-stabilized-ppo-teacher.md`;
  - `deep-research-report-13-hierarchical-dirichlet-rescue.md`.
- `stage0_audit/` contains the Stage 0 methodology, feature-set construction, model-selection scripts, and overnight PPO pipeline.
- `stage0_audit/Stage0_Teacher_Export_Guide.md` describes the post-selection hidden-state/action/context export for Joseph.
- `stage0_audit/feature_sets/` contains corrected model-ready datasets for:
  - `filtered_with_gru`;
  - `interpretable_no_gru`.
- `stage0_audit/model_runs/` contains walk-forward training and evaluation outputs.

## Current Methodology

1. Select the frozen PPO teacher by anchored walk-forward validation, not by frozen test.
2. Compare at least two compatible feature sets before Stage 1:
   - filtered features with GRU-derived signals;
   - interpretable no-GRU ablation.
3. Run the frozen test period, `2022-01-03` to `2023-02-28`, once after model selection is fixed.
4. Export policy-branch hidden states, actions, realized returns, dates, and aligned market context for Joseph's windowed VQ-VAE work.
5. In parallel with Stage 1 discovery, train Stage 0.1 stabilized PPO teachers with explicit stock+cash portfolio weights and smoother execution.
6. For Stage 0.1 hierarchy rescue, use the hypothesis ledger in `METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md`: width/feature controls first, then root split cash-vs-invested, then cascade/distribution/discovered-hierarchy extensions.
7. Use the proposal pipeline:
   - behavior discovery;
   - portfolio diagnostics;
   - finance-grounded labeling;
   - outcome analysis.

## Python Environment

Use the local TensorFlow environment for project scripts:

```powershell
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" stage0_audit\run_stage0_overnight.py
```

Stage 0.1 stabilized PPO smoke test:

```powershell
& "C:\Users\ivanp\anaconda3\envs\tensorflow\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_stabilized_ppo.yaml --smoke-test --force
```

## Legacy Folders

`latent_actions_experiments/` contains previous experiments around `base_macro`, SSL domain generalization, and behavior interpretability audit branches. These are archival/reference materials only. Use them for file-format examples when helpful, but do not treat their README files as the current research plan.
