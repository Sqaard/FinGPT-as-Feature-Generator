#!/usr/bin/env python
"""Run task.md text-feature group ablations on the R6c Stage0.1 agent.

The R6c launch package uses the DeepSeek compact text10 schema, so the task
groups are mapped onto those PPO-facing columns. Each ablation keeps the same
R6c market state and sets inactive text columns to zero in a group-specific raw
panel. Training uses the bundled Stage0.1 trainer, then this script evaluates
the trained fold_2021 model on the frozen OOS window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
R6C_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "r6c_stage0_1_text_baseline_20260530"
    / "rl_stage0_1_r6c_project"
)
BASE_CONFIG = R6C_ROOT / "configs" / "stage0_1_r6c_deepseek_v2_text.yaml"
RAW_TEXT10_PANEL = (
    R6C_ROOT
    / "artifacts"
    / "stage0_1"
    / "features"
    / "stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv"
)
OUT_ROOT = R6C_ROOT / "artifacts" / "r6c_text_feature_group_ablation"
REPO_OUT_ROOT = REPO_ROOT / "artifacts" / "r6c_text_feature_group_ablation"
VARIANT = "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"
BASELINE_FROZEN_CSV = (
    REPO_ROOT
    / "artifacts"
    / "r6c_stage0_1_text_baseline_20260530"
    / "comparison_summary_old_vs_r6c.csv"
)
RAW_TEXT10_FROZEN_CSV = (
    R6C_ROOT
    / "artifacts"
    / "stage0_1_text"
    / "r6c_deepseek_v2_text_frozen_oos"
    / "frozen_oos_results.csv"
)

TEXT_COLUMNS = [
    "text_alpha_direction",
    "text_downside_risk",
    "text_uncertainty",
    "text_macro_stress",
    "text_earnings_pressure",
    "text_balance_sheet_stress",
    "text_signal_confidence",
    "text_evidence_specificity",
    "text_numeric_evidence_density",
    "text_boilerplate_intensity",
]

STATIC_GROUPS: dict[str, dict[str, Any]] = {
    "risk_uncertainty": {
        "label": "Only risk / uncertainty",
        "features": [
            "text_downside_risk",
            "text_uncertainty",
            "text_balance_sheet_stress",
            "text_boilerplate_intensity",
        ],
    },
    "sentiment_price": {
        "label": "Only sentiment / price impact",
        "features": [
            "text_alpha_direction",
            "text_signal_confidence",
            "text_evidence_specificity",
        ],
    },
    "forward_earnings": {
        "label": "Only forward-looking / earnings guidance",
        "features": [
            "text_earnings_pressure",
            "text_numeric_evidence_density",
            "text_signal_confidence",
        ],
    },
    "macro_conditions": {
        "label": "Only macro financial conditions",
        "features": [
            "text_macro_stress",
            "text_uncertainty",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6c-root", type=Path, default=R6C_ROOT)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--raw-panel", type=Path, default=RAW_TEXT10_PANEL)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--repo-out-root", type=Path, default=REPO_OUT_ROOT)
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--folds", nargs="+", default=["fold_2021"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--force-panels", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-frozen-eval", action="store_true")
    return parser.parse_args()


def rel_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def check_data_ready(raw_panel: Path, base_config: Path) -> dict[str, Any]:
    if not raw_panel.exists():
        return {"ready": False, "reason": f"missing raw panel: {raw_panel}"}
    if not base_config.exists():
        return {"ready": False, "reason": f"missing base config: {base_config}"}
    header = list(pd.read_csv(raw_panel, nrows=0).columns)
    missing_text = [col for col in TEXT_COLUMNS if col not in header]
    return {
        "ready": not missing_text,
        "raw_panel": str(raw_panel),
        "base_config": str(base_config),
        "missing_text_features": missing_text,
        "text_feature_count": len([col for col in TEXT_COLUMNS if col in header]),
    }


def train_correlations(raw_panel: Path, train_start: str, train_end: str) -> pd.DataFrame:
    columns = ["date", "tic", "close", *TEXT_COLUMNS]
    df = pd.read_csv(raw_panel, usecols=columns, parse_dates=["date"])
    df = df.sort_values(["tic", "date"])
    df["next_return"] = df.groupby("tic")["close"].transform(lambda values: values.shift(-1) / values - 1.0)
    train = df[(df["date"] >= pd.Timestamp(train_start)) & (df["date"] <= pd.Timestamp(train_end))]
    rows = []
    for col in TEXT_COLUMNS:
        corr = train[col].corr(train["next_return"])
        rows.append(
            {
                "feature": col,
                "train_next_day_return_corr": corr,
                "abs_train_next_day_return_corr": abs(corr) if pd.notna(corr) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_train_next_day_return_corr", ascending=False)


def build_groups(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    folds = pd.read_csv(args.r6c_root / config["walk_forward"]["folds_csv"])
    fold = folds[folds["fold"].eq(args.folds[-1])].iloc[0]
    corr = train_correlations(args.raw_panel, str(fold["train_start"]), str(fold["train_end_inclusive"]))
    groups = {key: dict(value) for key, value in STATIC_GROUPS.items()}
    groups["topk_train_correlation"] = {
        "label": f"Top-{args.top_k} train-correlation features",
        "features": corr.head(args.top_k)["feature"].tolist(),
    }
    return groups, corr


def ensure_group_panel(args: argparse.Namespace, group_key: str, group: dict[str, Any]) -> Path:
    panel_path = args.out_root / "raw_panels" / f"stage0_1_weight_features_raw_{group_key}.csv"
    if panel_path.exists() and not args.force_panels:
        return panel_path
    df = pd.read_csv(args.raw_panel)
    active = set(group["features"])
    inactive = [col for col in TEXT_COLUMNS if col not in active]
    for col in inactive:
        df[col] = 0.0
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(panel_path, index=False)
    return panel_path


def ensure_group_config(args: argparse.Namespace, group_key: str, group: dict[str, Any], panel_path: Path) -> Path:
    config = load_config(args.base_config)
    config["experiment"]["name"] = f"stage0_1_r6c_text_ablation_{group_key}"
    config["data"]["feature_set"] = f"stage0_1_weight_features_r6c_text_ablation_{group_key}"
    config["data"]["raw_features_csv"] = rel_to_project(panel_path, args.r6c_root)
    config["data"]["model_ready_csv"] = (
        f"artifacts/r6c_text_feature_group_ablation/features/{group_key}/"
        "stage0_1_weight_features_model_ready.csv"
    )
    config["data"]["transform_stats_csv"] = (
        f"artifacts/r6c_text_feature_group_ablation/features/{group_key}/"
        "stage0_1_weight_feature_transform_stats.csv"
    )
    config["output"]["root_dir"] = "artifacts/r6c_text_feature_group_ablation/runs"
    config["output"]["run_name"] = group_key
    config["output"]["skip_completed"] = True
    config["text_features"]["ablation_group"] = group_key
    config["text_features"]["active_columns"] = list(group["features"])
    config["text_features"]["inactive_columns_zeroed"] = [col for col in TEXT_COLUMNS if col not in group["features"]]
    config_path = args.out_root / "configs" / f"stage0_1_r6c_text_ablation_{group_key}.yaml"
    write_yaml(config_path, config)
    return config_path


def run_training(args: argparse.Namespace, group_key: str, config_path: Path) -> None:
    run_root = args.out_root / "runs" / group_key
    completed = [
        run_root / args.variant / fold / "model.zip" for fold in args.folds
    ] + [
        run_root / args.variant / fold / "validation_summary.csv" for fold in args.folds
    ]
    if all(path.exists() for path in completed) and not args.force_train:
        print(f"Using existing R6c run for {group_key}")
        return
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{group_key}.log"
    cmd = [
        sys.executable,
        "-m",
        "src.ppo.stage0_1_train",
        "--config",
        rel_to_project(config_path, args.r6c_root),
        "--variants",
        args.variant,
        "--folds",
        *args.folds,
    ]
    if args.force_train:
        cmd.append("--force")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n\n=== Running " + " ".join(cmd) + " ===\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=args.r6c_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate_frozen_fold(args: argparse.Namespace, group_key: str, config_path: Path, fold_id: str) -> Path:
    out_dir = args.out_root / "frozen_oos" / group_key / args.variant / fold_id
    summary_path = out_dir / "frozen_test_summary.csv"
    if summary_path.exists() and not args.force_train:
        return summary_path

    sys.path.insert(0, str(args.r6c_root))
    from src.data.stage0_1_normalization import prepare_fold_scaled_features  # type: ignore
    from src.ppo.instrumented_ppo import InstrumentedPPO  # type: ignore
    from src.ppo.stage0_1_train import evaluate_model, load_yaml  # type: ignore
    from src.ppo.stage0_1_weight_env import load_weight_panel  # type: ignore

    config = load_yaml(config_path)
    folds = pd.read_csv(args.r6c_root / config["walk_forward"]["folds_csv"])
    fold = folds[folds["fold"].eq(fold_id)].iloc[0]
    run_dir = args.out_root / "runs" / group_key / args.variant / fold_id
    model_path = run_dir / "model.zip"
    metadata_path = run_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing trained model/metadata for {group_key} {fold_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_end_exclusive = pd.Timestamp(fold["frozen_test_end_exclusive"])
    frozen_end = (frozen_end_exclusive - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_csv = Path(config["data"]["raw_features_csv"])
    if not raw_csv.is_absolute():
        raw_csv = args.r6c_root / raw_csv
    normalization_cfg = config.get("normalization", {})
    frozen_feature_info = prepare_fold_scaled_features(
        raw_csv=raw_csv,
        out_dir=args.out_root / "frozen_oos_feature_scalers" / group_key,
        fold_id=fold_id,
        train_start=str(fold["train_start"]),
        train_end=str(fold["train_end_inclusive"]),
        validation_end=frozen_end,
        lower_quantile=float(normalization_cfg.get("lower_quantile", 0.01)),
        upper_quantile=float(normalization_cfg.get("upper_quantile", 0.99)),
        force=args.force_train,
    )
    model_ready_csv = Path(frozen_feature_info["model_ready_csv"])
    panel = load_weight_panel(
        model_ready_csv,
        str(fold["frozen_test_start"]),
        frozen_end,
    )
    model = InstrumentedPPO.load(str(model_path), device="cpu")
    summary = evaluate_model(model, panel, config, metadata["variant"], out_dir, "frozen_test")
    summary.update({"group": group_key, "fold": fold_id, "variant": args.variant, "model_path": str(model_path)})
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary_path


def collect_rows(args: argparse.Namespace, groups: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if BASELINE_FROZEN_CSV.exists():
        base = pd.read_csv(BASELINE_FROZEN_CSV)
        base = base[base["strategy"].eq("R6c Stage0.1 baseline frozen test")]
        if not base.empty:
            row = base.iloc[0]
            rows.append(
                {
                    "strategy": "R6c baseline",
                    "feature_group": "baseline",
                    "active_features": "",
                    "fold": "fold_2021",
                    "return": float(row["total_return"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover_l1_mean": float(row["turnover_l1_mean"]),
                    "source_file": str(BASELINE_FROZEN_CSV.relative_to(REPO_ROOT)),
                }
            )
    if RAW_TEXT10_FROZEN_CSV.exists():
        raw = pd.read_csv(RAW_TEXT10_FROZEN_CSV)
        raw = raw[raw["fold"].eq("fold_2021")]
        if not raw.empty:
            row = raw.iloc[0]
            rows.append(
                {
                    "strategy": "R6c + all raw text10",
                    "feature_group": "all_text10",
                    "active_features": ",".join(TEXT_COLUMNS),
                    "fold": "fold_2021",
                    "return": float(row["return_pct"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover_l1_mean": float(row["turnover_l1_mean"]),
                    "source_file": str(RAW_TEXT10_FROZEN_CSV.relative_to(REPO_ROOT)),
                }
            )
    for group_key, group in groups.items():
        for fold_id in args.folds:
            summary_path = args.out_root / "frozen_oos" / group_key / args.variant / fold_id / "frozen_test_summary.csv"
            if not summary_path.exists():
                continue
            row = pd.read_csv(summary_path).iloc[0]
            rows.append(
                {
                    "strategy": group["label"],
                    "feature_group": group_key,
                    "active_features": ",".join(group["features"]),
                    "fold": fold_id,
                    "return": float(row["return_pct"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "turnover_l1_mean": float(row["turnover_l1_mean"]),
                    "source_file": str(summary_path.relative_to(REPO_ROOT)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base_rows = out[out["feature_group"].eq("baseline")]
    base_return = float(base_rows["return"].iloc[0]) if not base_rows.empty else 0.0
    base_sharpe = float(base_rows["sharpe"].iloc[0]) if not base_rows.empty else 0.0
    out["return_delta_vs_r6c_baseline"] = out["return"] - base_return
    out["sharpe_delta_vs_r6c_baseline"] = out["sharpe"] - base_sharpe

    def classify(row: pd.Series) -> str:
        if row["feature_group"] == "baseline":
            return "baseline"
        if row["return_delta_vs_r6c_baseline"] > 0 and row["sharpe_delta_vs_r6c_baseline"] > 0:
            return "useful"
        if row["return_delta_vs_r6c_baseline"] < -0.02 and row["sharpe_delta_vs_r6c_baseline"] < -0.15:
            return "harmful"
        return "noisy/mixed"

    out["interpretation"] = out.apply(classify, axis=1)
    return out


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(args: argparse.Namespace, data_ready: dict[str, Any], groups: dict[str, dict[str, Any]], corr: pd.DataFrame, rows: pd.DataFrame) -> None:
    lines = [
        "# R6c Text Feature Group Ablation Report",
        "",
        "## Data readiness",
        "",
        f"Prepared: `{data_ready['ready']}`.",
        f"Raw R6c panel: `{Path(data_ready['raw_panel']).relative_to(REPO_ROOT)}`.",
        f"Missing text features: `{data_ready['missing_text_features']}`.",
        "",
        "## Frozen OOS ablation table",
        "",
    ]
    if rows.empty:
        lines.append("_No completed frozen OOS rows yet._")
    else:
        lines.extend(
            [
                "| Strategy | Fold | Return | Sharpe | Max drawdown | Interpretation |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['strategy']} | {row['fold']} | {pct(float(row['return']))} | "
                f"{float(row['sharpe']):.4f} | {pct(float(row['max_drawdown']))} | {row['interpretation']} |"
            )
    lines.extend(["", "## Feature groups", ""])
    for group_key, group in groups.items():
        lines.append(f"- `{group_key}`: {group['label']} = `{', '.join(group['features'])}`")
    lines.extend(
        [
            "",
            "## Top-k selection method",
            "",
            "Top-k uses absolute Pearson correlation between each DeepSeek text feature and next-day same-ticker return on the selected fold's train window.",
            "",
            "| feature | train_next_day_return_corr | abs_train_next_day_return_corr |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in corr.iterrows():
        lines.append(
            f"| {row['feature']} | {float(row['train_next_day_return_corr']):.6g} | "
            f"{float(row['abs_train_next_day_return_corr']):.6g} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if rows.empty:
        lines.append("Training/evaluation has not produced completed rows yet.")
    else:
        useful = rows[rows["interpretation"].eq("useful")]["strategy"].tolist()
        noisy = rows[rows["interpretation"].eq("noisy/mixed")]["strategy"].tolist()
        harmful = rows[rows["interpretation"].eq("harmful")]["strategy"].tolist()
        lines.append(f"Useful groups: {', '.join(useful) if useful else 'none'}.")
        lines.append(f"Noisy or mixed groups: {', '.join(noisy) if noisy else 'none'}.")
        lines.append(f"Harmful groups: {', '.join(harmful) if harmful else 'none'}.")
        lines.append("")
        lines.append(
            "This R6c rerun uses the newer DeepSeek compact text schema, so group names match task.md semantically but not one-to-one with the old Mistral feature columns."
        )
    args.repo_out_root.mkdir(parents=True, exist_ok=True)
    (args.repo_out_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(args: argparse.Namespace, data_ready: dict[str, Any], groups: dict[str, dict[str, Any]], corr: pd.DataFrame, rows: pd.DataFrame) -> None:
    args.repo_out_root.mkdir(parents=True, exist_ok=True)
    results_dir = args.repo_out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(results_dir / "train_text_feature_correlations.csv", index=False)
    pd.DataFrame(
        [
            {"feature_group": key, "label": value["label"], "active_features": ",".join(value["features"])}
            for key, value in groups.items()
        ]
    ).to_csv(results_dir / "feature_groups.csv", index=False)
    rows.to_csv(results_dir / "r6c_ablation_summary.csv", index=False)
    write_report(args, data_ready, groups, corr, rows)
    write_json(
        args.repo_out_root / "manifest.json",
        {
            "status": "completed" if data_ready["ready"] and not rows.empty else "incomplete",
            "data_ready": data_ready,
            "r6c_root": str(args.r6c_root),
            "variant": args.variant,
            "folds": args.folds,
            "groups": groups,
            "result_files": {
                "summary": str((results_dir / "r6c_ablation_summary.csv").relative_to(REPO_ROOT)),
                "report": str((args.repo_out_root / "final_report.md").relative_to(REPO_ROOT)),
            },
        },
    )


def main() -> None:
    args = parse_args()
    args.r6c_root = args.r6c_root.resolve()
    args.base_config = args.base_config.resolve()
    args.raw_panel = args.raw_panel.resolve()
    args.out_root = args.out_root.resolve()
    args.repo_out_root = args.repo_out_root.resolve()

    data_ready = check_data_ready(args.raw_panel, args.base_config)
    if not data_ready["ready"]:
        write_outputs(args, data_ready, {}, pd.DataFrame(), pd.DataFrame())
        raise RuntimeError(data_ready)

    base_config = load_config(args.base_config)
    groups, corr = build_groups(args, base_config)

    for group_key, group in groups.items():
        print(f"Preparing R6c ablation group: {group_key}")
        panel_path = ensure_group_panel(args, group_key, group)
        config_path = ensure_group_config(args, group_key, group, panel_path)
        if not args.skip_training:
            print(f"Training R6c group: {group_key}")
            run_training(args, group_key, config_path)
        if not args.skip_frozen_eval:
            for fold_id in args.folds:
                print(f"Evaluating frozen OOS: {group_key} {fold_id}")
                evaluate_frozen_fold(args, group_key, config_path, fold_id)

    rows = collect_rows(args, groups)
    write_outputs(args, data_ready, groups, corr, rows)
    print(json.dumps({"report": str(args.repo_out_root / "final_report.md"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
