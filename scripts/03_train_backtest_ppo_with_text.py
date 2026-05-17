#!/usr/bin/env python
"""
Train and backtest PPO with the 10 text features merged into the base panel.

This script is intentionally a thin wrapper around the existing RL project
stage0 PPO code. It does not redefine the trading environment; it reuses the
same environment, reward, date splits, and PPO config as the benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = TOY_ROOT / "artifacts" / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv"
DEFAULT_RL_PROJECT = TOY_ROOT / "rl_stage0_project"
TEXT_FEATURE_COLUMNS = [
    "text_relevance_to_portfolio",
    "text_sentiment_direction",
    "text_price_impact_direction",
    "text_risk_intensity",
    "text_uncertainty_intensity",
    "text_opportunity_intensity",
    "text_forward_looking_intensity",
    "text_earnings_guidance_impact",
    "text_macro_financial_conditions_impact",
    "text_company_event_risk_impact",
]
NON_FEATURE_COLUMNS = {"date", "tic", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="Merged base panel with text features.")
    parser.add_argument(
        "--rl-project",
        type=Path,
        default=DEFAULT_RL_PROJECT,
        help="Path to the RL benchmark project containing stage0_audit/stage0_model_pipeline.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=TOY_ROOT / "artifacts" / "ppo_with_text_run")
    parser.add_argument("--selected-config", default="custom_custom")
    parser.add_argument("--timesteps", type=int, default=350_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--turbulence-threshold", type=float, default=100.0)
    parser.add_argument("--train-start", default="2010-01-04")
    parser.add_argument("--train-end", default="2021-09-30")
    parser.add_argument("--validation-start", default="2021-10-01")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-03")
    parser.add_argument("--test-end", default="2023-02-28")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if a model already exists.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a PPO checkpoint, or 'latest' to resume from the newest checkpoint in output-dir/checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a manifest without importing FinRL or training PPO.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def resolve_resume_checkpoint(resume_arg: str | None, checkpoints_dir: Path) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg.lower() == "latest":
        checkpoints = sorted(
            checkpoints_dir.glob("ppo_with_text_checkpoint_*_steps.zip"),
            key=checkpoint_step,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        return checkpoints[-1]
    checkpoint = Path(resume_arg)
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    checkpoint_step(checkpoint)
    return checkpoint


def load_stage0_pipeline(rl_project: Path):
    stage0_dir = rl_project / "stage0_audit"
    if not stage0_dir.exists():
        raise FileNotFoundError(
            f"Cannot find stage0_audit at {stage0_dir}. "
            "Pass --rl-project pointing to data_RLagent_for_Joseph."
        )
    sys.path.insert(0, str(stage0_dir))
    import stage0_model_pipeline as stage0  # type: ignore

    return stage0


def validate_panel_header(panel: Path) -> dict[str, Any]:
    if not panel.exists():
        raise FileNotFoundError(f"Panel not found: {panel}")
    header = list(pd.read_csv(panel, nrows=0).columns)
    missing = [col for col in TEXT_FEATURE_COLUMNS if col not in header]
    date_col = "date" if "date" in header else ("Date" if "Date" in header else None)
    ticker_col = "tic" if "tic" in header else ("ticker" if "ticker" in header else None)
    return {
        "panel": str(panel),
        "column_count": len(header),
        "date_column": date_col,
        "ticker_column": ticker_col,
        "text_feature_columns": [col for col in TEXT_FEATURE_COLUMNS if col in header],
        "missing_text_feature_columns": missing,
    }


def prepare_numeric_feature_columns(
    df: pd.DataFrame,
    raw_feature_columns: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Keep only numeric state features for FinRL observations.

    The merged panel intentionally preserves provenance helper columns such as
    date_available/source timestamps. They are useful for auditing, but FinRL
    StockTradingEnv expects every tech_indicator_list value to be numeric.
    """
    feature_columns: list[str] = []
    coerced_numeric_columns: list[str] = []
    excluded_non_numeric_columns: list[dict[str, Any]] = []

    for col in raw_feature_columns:
        if col not in df.columns:
            continue

        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            excluded_non_numeric_columns.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "reason": "datetime/provenance column is not a PPO state feature",
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            feature_columns.append(col)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_missing = int(series.notna().sum())
        numeric_parseable = int(converted.notna().sum())

        if non_missing > 0 and numeric_parseable == non_missing:
            df[col] = converted
            feature_columns.append(col)
            coerced_numeric_columns.append(col)
            continue

        excluded_non_numeric_columns.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "reason": "non-numeric helper/provenance column",
                "non_missing": non_missing,
                "numeric_parseable": numeric_parseable,
                "sample_values": series.dropna().astype(str).head(3).tolist(),
            }
        )

    if feature_columns:
        df.loc[:, feature_columns] = (
            df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
        )

    audit = {
        "raw_feature_column_count": len(raw_feature_columns),
        "numeric_feature_column_count": len(feature_columns),
        "coerced_numeric_columns": coerced_numeric_columns,
        "excluded_non_numeric_columns": excluded_non_numeric_columns,
    }
    return feature_columns, audit


def dry_run(args: argparse.Namespace) -> None:
    header_info = validate_panel_header(args.panel)
    sample = pd.read_csv(args.panel, nrows=1000)
    raw_feature_columns = [c for c in sample.columns if c not in NON_FEATURE_COLUMNS]
    feature_columns, feature_audit = prepare_numeric_feature_columns(sample, raw_feature_columns)
    payload = {
        "status": "dry_run_ok" if not header_info["missing_text_feature_columns"] else "dry_run_missing_text_features",
        "panel_header": header_info,
        "sample_rows_checked": len(sample),
        "numeric_feature_column_count": len(feature_columns),
        "feature_audit_sample": feature_audit,
        "rl_project_exists": args.rl_project.exists(),
        "stage0_audit_exists": (args.rl_project / "stage0_audit").exists(),
        "training_was_not_started": True,
    }
    write_json(args.output_dir / "manifest_dry_run.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def summary_for_comparison(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy": "PPO with text",
        "total_return": float(summary.get("return_pct", 0.0)) / 100.0,
        "sharpe": float(summary.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)


def save_actions(actions: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(actions, "to_csv"):
        actions.to_csv(output_path, index=False)
    else:
        output_path.with_suffix(".json").write_text(json.dumps(actions, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run(args)
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    header_info = validate_panel_header(args.panel)
    if header_info["missing_text_feature_columns"]:
        raise ValueError(f"Panel is missing text features: {header_info['missing_text_feature_columns']}")

    stage0 = load_stage0_pipeline(args.rl_project)
    deps = stage0.require_training_dependencies()
    drl_agent_cls = deps["DRLAgent"]
    stock_env_cls = deps["StockTradingEnv"]
    checkpoint_cls = deps["CheckpointCallback"]
    torch = deps["torch"]
    ppo_cls = deps["PPO"]
    backtest_stats = deps["backtest_stats"]

    df = stage0.load_feature_panel(str(args.panel))
    raw_feature_columns = stage0.infer_feature_columns(df)
    feature_columns, feature_audit = prepare_numeric_feature_columns(df, raw_feature_columns)
    text_features_used = [col for col in TEXT_FEATURE_COLUMNS if col in feature_columns]
    if len(text_features_used) != len(TEXT_FEATURE_COLUMNS):
        missing_numeric_text_features = [col for col in TEXT_FEATURE_COLUMNS if col not in text_features_used]
        raise ValueError(f"Text features are present but not numeric: {missing_numeric_text_features}")
    write_json(args.output_dir / "feature_column_audit.json", feature_audit)

    reward_name = stage0.config_reward_name(args.selected_config)
    env_cls = stage0.make_env_class(stock_env_cls, reward_name)
    policy_kwargs = stage0.config_policy_kwargs(args.selected_config, torch)
    model_kwargs = dict(stage0.DEFAULT_PPO_KWARGS)
    if args.learning_rate is not None:
        model_kwargs["learning_rate"] = args.learning_rate
    if args.n_steps is not None:
        model_kwargs["n_steps"] = args.n_steps
    if args.batch_size is not None:
        model_kwargs["batch_size"] = args.batch_size
    if args.ent_coef is not None:
        model_kwargs["ent_coef"] = args.ent_coef
    if args.target_kl is not None:
        model_kwargs["target_kl"] = args.target_kl

    model_path = args.output_dir / "checkpoints" / f"ppo_with_text_{args.selected_config}_{args.timesteps}_steps.zip"
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir / "checkpoints")
    if model_path.exists() and not args.force_train:
        print(f"Using existing model: {model_path}")
        model = ppo_cls.load(str(model_path), device="cpu")
    else:
        train_df = stage0.split_by_date(df, args.train_start, args.train_end)
        train_env_obj = stage0.build_env(
            env_cls,
            train_df,
            feature_columns,
            args.initial_amount,
            args.turbulence_threshold,
        )
        train_env, _ = train_env_obj.get_sb_env()
        if resume_checkpoint is not None:
            completed_steps = checkpoint_step(resume_checkpoint)
            remaining_steps = max(args.timesteps - completed_steps, 0)
            print(f"Resuming from {resume_checkpoint} at {completed_steps} steps; remaining target steps: {remaining_steps}")
            model = ppo_cls.load(str(resume_checkpoint), env=train_env, device="cpu")
            reset_num_timesteps = False
            learn_timesteps = remaining_steps
        else:
            agent = drl_agent_cls(env=train_env)
            model = agent.get_model("ppo", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
            reset_num_timesteps = True
            learn_timesteps = args.timesteps
        callback = checkpoint_cls(
            save_freq=args.save_freq,
            save_path=str(args.output_dir / "checkpoints"),
            name_prefix="ppo_with_text_checkpoint",
        )
        if learn_timesteps > 0:
            model.learn(
                total_timesteps=learn_timesteps,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,
            )
        else:
            print("Checkpoint already reached or exceeded requested --timesteps; saving/evaluating as final model.")
        model.save(str(model_path))
        print(f"Saved model: {model_path}")

    def evaluate_period(period_name: str, start: str, end: str) -> dict[str, Any]:
        period_df = stage0.split_by_date(df, start, end)
        env_obj = stage0.build_env(
            env_cls,
            period_df,
            feature_columns,
            args.initial_amount,
            args.turbulence_threshold,
        )
        account_value, actions = drl_agent_cls.DRL_prediction(model=model, environment=env_obj)
        curve, summary = stage0.evaluate_account_value(
            account_value,
            strategy="PPO with text",
            period=period_name,
            strategy_type="RL",
            initial_amount=args.initial_amount,
            backtest_stats=backtest_stats,
            extra={
                "feature_set": "mistral_text_10",
                "selected_config": args.selected_config,
                "text_feature_count": len(text_features_used),
            },
        )
        curve.to_csv(args.output_dir / "results" / f"ppo_with_text_{period_name}_curve.csv", index=False)
        pd.DataFrame([summary]).to_csv(
            args.output_dir / "results" / f"ppo_with_text_{period_name}_summary.csv",
            index=False,
        )
        save_actions(actions, args.output_dir / "results" / f"ppo_with_text_{period_name}_actions.csv")
        return summary

    validation_summary = evaluate_period("validation", args.validation_start, args.validation_end)
    test_summary = evaluate_period("test", args.test_start, args.test_end)
    summary_for_comparison(
        test_summary,
        args.output_dir / "results" / "ppo_with_text_summary_for_comparison.csv",
    )

    manifest = {
        "status": "completed",
        "panel": str(args.panel),
        "model_path": str(model_path),
        "selected_config": args.selected_config,
        "timesteps": args.timesteps,
        "model_kwargs": model_kwargs,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "feature_column_count": len(feature_columns),
        "raw_feature_column_count": len(raw_feature_columns),
        "text_features_used": text_features_used,
        "excluded_non_numeric_feature_columns": feature_audit["excluded_non_numeric_columns"],
        "train_window": [args.train_start, args.train_end],
        "validation_window": [args.validation_start, args.validation_end],
        "test_window": [args.test_start, args.test_end],
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "results_dir": str(args.output_dir / "results"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
