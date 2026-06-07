# Active Project Structure

This project now uses a clean active implementation layer while preserving old Stage 0 and legacy experiments as references.

## Active Folders

```text
configs/        stage configs and experiment contracts
src/            active reusable implementation code
artifacts/      generated outputs by stage
notebooks/      manual inspection notebooks only
reports/        human-readable plans and stage reports
```

## Source Modules

```text
src/data/        data loading, feature contracts, split utilities
src/ppo/         stabilized PPO, instrumented PPO, rollout extraction
src/ssl/         VQ-VAE, transformer VQ-VAE, primitive assignment
src/diagnostics/ action, portfolio, hidden-state, PPO, and risk diagnostics
src/labeling/    finance labels and market mechanism bank
src/causal/      one-step and sequential intervention audits
src/adapter/     primitive-aware adapter experiments
src/evaluation/  walk-forward, bootstrap, purged CV, reporting helpers
```

## Legacy / Completed Areas

`stage0_audit/` remains the completed Stage 0 audit and final-teacher export area. Do not move it until all reports and Joseph hand-off files are stable.

`latent_actions_experiments/` remains legacy/reference-only.

## Current Stage Routing

```text
Stage 0   completed in stage0_audit/ and artifacts/stage0/
Stage 0.1 new stabilized PPO work goes to src/ppo/ + configs/stage0_1_stabilized_ppo.yaml + artifacts/stage0_1/
Stage 1   primitive discovery goes to src/ssl/ + src/diagnostics/ + configs/stage1_*.yaml + artifacts/stage1/
```

## Rule

New code should go into `src/`. New generated files should go into `artifacts/`. `stage0_audit/` should only receive small fixes needed for already-created Stage 0 artifacts.
