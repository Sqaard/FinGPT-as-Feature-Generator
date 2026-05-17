# Local Stage0 PPO Code

This folder makes the toy package portable.

It contains the minimal Stage0 PPO training/evaluation code copied from the
original RL project:

- `stage0_audit/stage0_model_pipeline.py`
- `stage0_audit/stage0_methodology.py`

The scripts still require a Python environment with FinRL, Stable-Baselines3,
PyTorch, pandas, numpy, and pyfolio installed. They no longer require access to
the original absolute RL project path.
