from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from stage0_methodology import (
        DEFAULT_ANCHORED_FOLDS,
        FROZEN_TEST_END,
        FROZEN_TEST_START,
        assert_panel_integrity,
        assert_threshold_policy_consistent,
        final_acceptance_gate,
        walk_forward_selection_score,
    )
except ImportError:
    from .stage0_methodology import (
        DEFAULT_ANCHORED_FOLDS,
        FROZEN_TEST_END,
        FROZEN_TEST_START,
        assert_panel_integrity,
        assert_threshold_policy_consistent,
        final_acceptance_gate,
        walk_forward_selection_score,
    )


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "stage0_audit" / "feature_sets"
OUT_DIR = ROOT / "stage0_audit" / "model_runs"

DEFAULT_PPO_KWARGS = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

ALL_CONFIGS = [
    "finrl_finrl",
    "finrl_custom",
    "zhang_finrl",
    "zhang_custom",
    "custom_finrl",
    "custom_custom",
]


def require_training_dependencies() -> dict[str, object]:
    try:
        from finrl.agents.stablebaselines3.models import DRLAgent
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        from finrl.meta.preprocessor.preprocessors import data_split
        from finrl.plot import backtest_stats, get_baseline
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Run this script in the same Python "
            "environment/kernel that can execute the FinRL notebooks. Original import "
            f"error: {exc}"
        ) from exc

    return {
        "DRLAgent": DRLAgent,
        "StockTradingEnv": StockTradingEnv,
        "data_split": data_split,
        "backtest_stats": backtest_stats,
        "get_baseline": get_baseline,
        "PPO": PPO,
        "CheckpointCallback": CheckpointCallback,
        "torch": torch,
    }


def extract_metric(stats: object, metric_name: str) -> float:
    if isinstance(stats, pd.Series):
        return pd.to_numeric(stats.get(metric_name, np.nan), errors="coerce")
    if isinstance(stats, dict):
        return pd.to_numeric(stats.get(metric_name, np.nan), errors="coerce")
    if isinstance(stats, pd.DataFrame):
        if metric_name in stats.index:
            value = stats.loc[metric_name]
            return pd.to_numeric(value.iloc[0] if isinstance(value, pd.Series) else value, errors="coerce")
        if metric_name in stats.columns:
            return pd.to_numeric(stats[metric_name].iloc[0], errors="coerce")
    return np.nan


def stats_to_dict(stats: object) -> dict[str, float]:
    return {
        "annual_return": extract_metric(stats, "Annual return"),
        "cumulative_returns": extract_metric(stats, "Cumulative returns"),
        "annual_volatility": extract_metric(stats, "Annual volatility"),
        "sharpe_ratio": extract_metric(stats, "Sharpe ratio"),
        "calmar_ratio": extract_metric(stats, "Calmar ratio"),
        "stability": extract_metric(stats, "Stability"),
        "max_drawdown": extract_metric(stats, "Max drawdown"),
        "omega_ratio": extract_metric(stats, "Omega ratio"),
        "sortino_ratio": extract_metric(stats, "Sortino ratio"),
        "tail_ratio": extract_metric(stats, "Tail ratio"),
        "daily_value_at_risk": extract_metric(stats, "Daily value at risk"),
    }


def normalize_account_value(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out = out.reset_index()
        if "index" in out.columns:
            out = out.rename(columns={"index": "date"})
    value_col = "account_value" if "account_value" in out.columns else out.columns[1]
    out = out.rename(columns={value_col: "account_value"})
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", "account_value"]].sort_values("date").reset_index(drop=True)


def add_curve_features(df: pd.DataFrame, initial_amount: float) -> pd.DataFrame:
    out = normalize_account_value(df)
    out["daily_return"] = out["account_value"].pct_change().fillna(0.0)
    out["cumulative_return"] = out["account_value"] / initial_amount - 1.0
    out["running_max"] = out["account_value"].cummax()
    out["drawdown"] = out["account_value"] / out["running_max"] - 1.0
    return out


def evaluate_account_value(
    df_account_value: pd.DataFrame,
    strategy: str,
    period: str,
    strategy_type: str,
    initial_amount: float,
    backtest_stats: Callable[..., object],
    extra: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    curve = add_curve_features(df_account_value, initial_amount)
    stats = backtest_stats(account_value=curve)
    summary = {
        "period": period,
        "strategy": strategy,
        "strategy_type": strategy_type,
        "initial_value": curve["account_value"].iloc[0],
        "final_value": curve["account_value"].iloc[-1],
        "return_pct": (curve["account_value"].iloc[-1] / curve["account_value"].iloc[0] - 1.0) * 100.0,
        **stats_to_dict(stats),
    }
    if extra:
        summary.update(extra)
        for key, value in extra.items():
            curve[key] = value
    curve["period"] = period
    curve["strategy"] = strategy
    curve["strategy_type"] = strategy_type
    return curve, summary


def load_feature_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    assert_panel_integrity(df)
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def split_by_date(df: pd.DataFrame, start: str, end_inclusive: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_inclusive)
    out = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    # FinRL StockTradingEnv expects df.loc[day, :] to return the full cross-section
    # for one trading day, matching finrl.meta.preprocessor.preprocessors.data_split.
    out.index = out["date"].factorize()[0]
    return out


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"date", "tic", "close"}]


def make_env_class(base_env_cls: type, reward_name: str) -> type:
    if reward_name == "finrl":
        return base_env_cls

    class RewardEnv(base_env_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args,
            sigma_target: float = 0.15 / np.sqrt(252),
            lambda_param: float = 0.01,
            mu: float = 1.0,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.sigma_target = sigma_target
            self.lambda_param = lambda_param
            self.mu = mu
            self.returns_history: list[float] = []
            self.window_size = 20

        def reset(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.returns_history = []
            return super().reset(*args, **kwargs)

        def _get_total_asset(self) -> float:
            prices = np.asarray(self.state[1 : 1 + self.stock_dim], dtype=float)
            shares = np.asarray(self.state[1 + self.stock_dim : 1 + (2 * self.stock_dim)], dtype=float)
            cash = float(self.state[0])
            return float(cash + np.dot(prices, shares))

        def _get_scale_factor(self) -> float:
            hist = self.returns_history[-self.window_size :]
            current_sigma = np.std(hist) if len(hist) > 1 else 0.01
            current_sigma = max(float(current_sigma), 1e-4)
            return float(self.sigma_target / current_sigma)

        def step(self, actions):  # noqa: ANN001
            asset_before = self._get_total_asset()
            scale_factor = self._get_scale_factor()
            step_result = super().step(actions)

            if len(step_result) == 5:
                state, base_reward, done, truncated, info = step_result
            else:
                state, base_reward, done, info = step_result
                truncated = False

            asset_after = self._get_total_asset()
            port_return = (asset_after - asset_before) / asset_before if asset_before > 0 else 0.0

            if reward_name == "zhang":
                reward = self.mu * scale_factor * port_return * asset_before * self.reward_scaling
            elif reward_name == "custom":
                utility = port_return - self.lambda_param * (port_return**2)
                reward = scale_factor * utility * asset_before * self.reward_scaling
            else:
                reward = base_reward

            self.returns_history.append(float(port_return))
            if len(self.returns_history) > self.window_size:
                self.returns_history.pop(0)

            if len(step_result) == 5:
                return state, reward, done, truncated, info
            return state, reward, done, info

    RewardEnv.__name__ = f"{reward_name.capitalize()}StockTradingEnv"
    return RewardEnv


def base_env_kwargs(
    stock_dimension: int,
    feature_columns: list[str],
    initial_amount: float,
    threshold: float | None,
) -> dict[str, object]:
    state_space = 1 + (2 * stock_dimension) + (len(feature_columns) * stock_dimension)
    kwargs: dict[str, object] = {
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": feature_columns,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 10000,
    }
    if threshold is not None:
        kwargs["turbulence_threshold"] = threshold
        kwargs["risk_indicator_col"] = "turbulence"
    return kwargs


def build_env(
    env_cls: type,
    data: pd.DataFrame,
    feature_columns: list[str],
    initial_amount: float,
    threshold: float | None,
):
    stock_dimension = data["tic"].nunique()
    return env_cls(
        df=data,
        **base_env_kwargs(
            stock_dimension=stock_dimension,
            feature_columns=feature_columns,
            initial_amount=initial_amount,
            threshold=threshold,
        ),
    )


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps(?:_final)?\.zip$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def build_buy_and_hold(df_long: pd.DataFrame, initial_amount: float, buy_cost_pct: float = 0.001) -> pd.DataFrame:
    prices = df_long.pivot_table(index="date", columns="tic", values="close", aggfunc="last")
    prices = prices.sort_index().ffill()
    first_prices = prices.iloc[0].dropna()
    prices = prices[first_prices.index]
    investable = initial_amount * (1.0 - buy_cost_pct)
    shares = (investable / len(first_prices)) / first_prices
    portfolio = (prices * shares).sum(axis=1)
    return pd.DataFrame({"date": prices.index, "account_value": portfolio.values})


def build_dji_baseline(
    get_baseline: Callable[..., pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_amount: float,
) -> pd.DataFrame:
    df = get_baseline(ticker="^DJI", start=start, end=end)
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    first_close = df["close"].iloc[0]
    df["account_value"] = initial_amount * df["close"] / first_close
    return df[["date", "account_value"]]


def config_reward_name(config_name: str) -> str:
    if config_name.startswith("zhang_"):
        return "zhang"
    if config_name.startswith("custom_"):
        return "custom"
    return "finrl"


def config_policy_kwargs(config_name: str, torch_module: object) -> dict[str, object] | None:
    if not config_name.endswith("_custom"):
        return None
    return {
        "net_arch": {
            "pi": [256, 128, 64],
            "vf": [256, 128, 64],
        },
        "activation_fn": torch_module.nn.ReLU,
        "ortho_init": True,
    }


def train_walk_forward(args: argparse.Namespace) -> None:
    deps = require_training_dependencies()
    DRLAgent = deps["DRLAgent"]
    StockTradingEnv = deps["StockTradingEnv"]
    CheckpointCallback = deps["CheckpointCallback"]
    PPO = deps["PPO"]
    torch = deps["torch"]

    df = load_feature_panel(args.data)
    feature_columns = infer_feature_columns(df)
    output_dir = args.output_dir / args.feature_set / "walk_forward"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_threshold = args.turbulence_threshold
    validation_threshold = args.turbulence_threshold
    test_threshold = args.turbulence_threshold
    assert_threshold_policy_consistent(train_threshold, validation_threshold, test_threshold)

    metadata = {
        "feature_set": args.feature_set,
        "data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
        "feature_columns": feature_columns,
        "configs": args.configs,
        "timesteps": args.timesteps,
        "save_freq": args.save_freq,
        "threshold": args.turbulence_threshold,
        "folds": [asdict(fold) for fold in DEFAULT_ANCHORED_FOLDS],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for fold in DEFAULT_ANCHORED_FOLDS:
        train_df = split_by_date(df, fold.train_start, fold.train_end_inclusive)
        for config_name in args.configs:
            reward_name = config_reward_name(config_name)
            env_cls = make_env_class(StockTradingEnv, reward_name)
            train_env_obj = build_env(
                env_cls,
                train_df,
                feature_columns,
                args.initial_amount,
                train_threshold,
            )
            train_env, _ = train_env_obj.get_sb_env()
            policy_kwargs = config_policy_kwargs(config_name, torch)

            checkpoint_dir = output_dir / fold.fold / config_name / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            final_path = checkpoint_dir / f"ppo_{config_name}_{args.timesteps}_steps.zip"
            if final_path.exists() and not getattr(args, "force_train", False):
                print(f"Skipping completed run: {final_path}")
                continue

            checkpoint_files = sorted(
                checkpoint_dir.glob(f"ppo_{config_name}_*_steps*.zip"),
                key=checkpoint_step,
            )
            checkpoint_files = [p for p in checkpoint_files if checkpoint_step(p) <= args.timesteps]

            if checkpoint_files and not getattr(args, "force_train", False):
                latest_checkpoint = checkpoint_files[-1]
                current_step = checkpoint_step(latest_checkpoint)
                if current_step >= args.timesteps:
                    print(f"Skipping completed run: {latest_checkpoint}")
                    continue
                print(f"Resuming {config_name} {fold.fold} from {latest_checkpoint} at {current_step} steps")
                model = PPO.load(str(latest_checkpoint), env=train_env)
                learn_timesteps = args.timesteps - current_step
                reset_num_timesteps = False
            else:
                agent = DRLAgent(env=train_env)
                model = agent.get_model(
                    "ppo",
                    model_kwargs=DEFAULT_PPO_KWARGS.copy(),
                    policy_kwargs=policy_kwargs,
                )
                learn_timesteps = args.timesteps
                reset_num_timesteps = True

            callback = CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"ppo_{config_name}",
            )
            model.learn(
                total_timesteps=learn_timesteps,
                tb_log_name=f"{args.feature_set}_{fold.fold}_{config_name}",
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
            )
            model.save(final_path)


def evaluate_walk_forward(args: argparse.Namespace) -> None:
    deps = require_training_dependencies()
    DRLAgent = deps["DRLAgent"]
    StockTradingEnv = deps["StockTradingEnv"]
    PPO = deps["PPO"]
    backtest_stats = deps["backtest_stats"]

    df = load_feature_panel(args.data)
    feature_columns = infer_feature_columns(df)
    output_dir = args.output_dir / args.feature_set / "walk_forward"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    curves = []

    for fold in DEFAULT_ANCHORED_FOLDS:
        validation_df = split_by_date(df, fold.validation_start, fold.validation_end_inclusive)
        for config_name in args.configs:
            reward_name = config_reward_name(config_name)
            env_cls = make_env_class(StockTradingEnv, reward_name)
            checkpoint_dir = output_dir / fold.fold / config_name / "checkpoints"
            checkpoint_files = sorted(checkpoint_dir.glob(f"ppo_{config_name}_*_steps*.zip"), key=checkpoint_step)
            if args.max_checkpoints:
                checkpoint_files = checkpoint_files[: args.max_checkpoints]

            for checkpoint in checkpoint_files:
                step = checkpoint_step(checkpoint)
                model = PPO.load(str(checkpoint))
                val_env = build_env(
                    env_cls,
                    validation_df,
                    feature_columns,
                    args.initial_amount,
                    args.turbulence_threshold,
                )
                account_value, actions = DRLAgent.DRL_prediction(model=model, environment=val_env)
                extra = {
                    "feature_set": args.feature_set,
                    "fold": fold.fold,
                    "config_name": config_name,
                    "checkpoint_step": step,
                    "model_path": str(checkpoint.relative_to(ROOT)),
                    "threshold": args.turbulence_threshold,
                }
                curve, summary = evaluate_account_value(
                    account_value,
                    strategy=f"{config_name}_{step}",
                    period="validation",
                    strategy_type="RL",
                    initial_amount=args.initial_amount,
                    backtest_stats=backtest_stats,
                    extra=extra,
                )
                curves.append(curve)
                summaries.append(summary)

                if args.save_actions:
                    actions_path = eval_dir / f"actions_{args.feature_set}_{fold.fold}_{config_name}_{step}.csv"
                    if isinstance(actions, pd.DataFrame):
                        actions.to_csv(actions_path, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(eval_dir / "walk_forward_validation_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            eval_dir / "walk_forward_validation_curves.csv", index=False
        )

    if not summary_df.empty:
        score_input = summary_df.rename(columns={"sharpe_ratio": "validation_sharpe"})
        scores = walk_forward_selection_score(
            score_input,
            sharpe_col="validation_sharpe",
            group_cols=("feature_set", "config_name", "checkpoint_step"),
        )
        scores.to_csv(eval_dir / "walk_forward_selection_scores.csv", index=False)


def final_test(args: argparse.Namespace) -> None:
    deps = require_training_dependencies()
    DRLAgent = deps["DRLAgent"]
    StockTradingEnv = deps["StockTradingEnv"]
    PPO = deps["PPO"]
    backtest_stats = deps["backtest_stats"]
    get_baseline = deps["get_baseline"]

    if not args.model_path:
        raise ValueError("--model-path is required for final-test mode.")

    df = load_feature_panel(args.data)
    feature_columns = infer_feature_columns(df)
    test_df = split_by_date(
        df,
        FROZEN_TEST_START.strftime("%Y-%m-%d"),
        (FROZEN_TEST_END - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    config_name = args.selected_config
    reward_name = config_reward_name(config_name)
    env_cls = make_env_class(StockTradingEnv, reward_name)
    model = PPO.load(str(args.model_path))

    test_env = build_env(
        env_cls,
        test_df,
        feature_columns,
        args.initial_amount,
        args.turbulence_threshold,
    )
    account_value, actions = DRLAgent.DRL_prediction(model=model, environment=test_env)

    output_dir = args.output_dir / args.feature_set / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    curves = []
    extra = {
        "feature_set": args.feature_set,
        "config_name": config_name,
        "model_path": str(Path(args.model_path)),
        "threshold": args.turbulence_threshold,
    }
    curve, summary = evaluate_account_value(
        account_value,
        strategy=config_name,
        period="frozen_test",
        strategy_type="RL",
        initial_amount=args.initial_amount,
        backtest_stats=backtest_stats,
        extra=extra,
    )
    curves.append(curve)
    summaries.append(summary)

    for name, baseline_df in [
        ("BuyAndHold_equal_weight", build_buy_and_hold(test_df, args.initial_amount)),
        (
            "DJI_baseline",
            build_dji_baseline(get_baseline, FROZEN_TEST_START, FROZEN_TEST_END, args.initial_amount),
        ),
    ]:
        baseline_curve, baseline_summary = evaluate_account_value(
            baseline_df,
            strategy=name,
            period="frozen_test",
            strategy_type="Benchmark",
            initial_amount=args.initial_amount,
            backtest_stats=backtest_stats,
            extra={"feature_set": args.feature_set, "config_name": name},
        )
        curves.append(baseline_curve)
        summaries.append(baseline_summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "final_test_summary.csv", index=False)
    final_acceptance_gate(summary_df).to_csv(output_dir / "final_acceptance_gate.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "final_test_curves.csv", index=False)
    if args.save_actions and isinstance(actions, pd.DataFrame):
        actions.to_csv(output_dir / "final_test_actions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 PPO walk-forward training and evaluation.")
    parser.add_argument(
        "mode",
        choices=["train-walk-forward", "evaluate-walk-forward", "final-test"],
    )
    parser.add_argument("--feature-set", default="filtered_with_gru")
    parser.add_argument(
        "--data",
        type=Path,
        default=FEATURE_DIR / "filtered_with_gru_model_ready.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=ALL_CONFIGS,
    )
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--turbulence-threshold", type=float, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--selected-config", default="finrl_finrl")
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.mode == "train-walk-forward":
        train_walk_forward(args)
    elif args.mode == "evaluate-walk-forward":
        evaluate_walk_forward(args)
    elif args.mode == "final-test":
        final_test(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
