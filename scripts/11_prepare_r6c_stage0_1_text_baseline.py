#!/usr/bin/env python
"""Prepare the R6c Stage0.1 baseline for text-feature PPO experiments.

The source RL project is treated as read-only. This script copies the minimal
Stage0.1 code/configs into this repository, builds a local raw feature panel
with DeepSeek v2 text features, extracts baseline metrics, and writes a launch
package for Huawei/local runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RL_ROOT = Path(r"C:\Users\ivanp\RL for Time-Series Forecasting\data_RLagent_for_Joseph")
R6C_VARIANT = "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "r6c_stage0_1_text_baseline_20260530"
TEXT_SCHEMA = REPO_ROOT / "feature_schema_deepseek_v2_ppo_compact.json"
TEXT_PANEL = REPO_ROOT / "artifacts" / "processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2.csv"
OLD_BENCHMARK = REPO_ROOT / "ppo_without_text_BENCHMARK" / "benchmark_summary.csv"
OLD_TEXT_COMPARISON = REPO_ROOT / "artifacts" / "ppo_text_vs_benchmark" / "results" / "comparison_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rl-root", type=Path, default=DEFAULT_SOURCE_RL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--text-panel", type=Path, default=TEXT_PANEL)
    parser.add_argument("--text-schema", type=Path, default=TEXT_SCHEMA)
    parser.add_argument("--no-zip", action="store_true", help="Skip launch-package zip creation.")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copytree_merge(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_required_stage0_project(source_rl_root: Path, local_project: Path) -> None:
    local_project.mkdir(parents=True, exist_ok=True)
    copytree_merge(source_rl_root / "src", local_project / "src")
    copytree_merge(source_rl_root / "configs", local_project / "configs")
    for name in ["README.md", "PROJECT_STRUCTURE.md", "METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md"]:
        src = source_rl_root / name
        if src.exists():
            shutil.copy2(src, local_project / name)
    readme = local_project / "README_R6C_TEXT_STAGING.md"
    readme.write_text(
        "\n".join(
            [
                "# R6c Stage0.1 Text Staging Project",
                "",
                "This is a local copy prepared from the external RL project.",
                "Do not edit the source RL repository for text-feature experiments.",
                "",
                "Primary launch config:",
                "",
                "`configs/stage0_1_r6c_deepseek_v2_text.yaml`",
                "",
                "Primary command:",
                "",
                "```powershell",
                '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants '
                + R6C_VARIANT,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_text_feature_columns(schema_path: Path) -> list[str]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    columns = [str(item["name"]) for item in features]
    if not columns:
        raise ValueError(f"No features found in {schema_path}")
    missing_prefix = [col for col in columns if not col.startswith("text_")]
    if missing_prefix:
        raise ValueError(f"Expected PPO-facing text_ columns, got: {missing_prefix}")
    return columns


def build_text_augmented_raw_panel(
    *,
    source_raw_panel: Path,
    text_panel: Path,
    output_raw_panel: Path,
    text_columns: list[str],
) -> dict[str, Any]:
    base = pd.read_csv(source_raw_panel)
    text = pd.read_csv(text_panel, usecols=["date", "tic", *text_columns])

    for frame in [base, text]:
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["tic"] = frame["tic"].astype(str).str.upper()

    duplicate_text_keys = int(text.duplicated(["date", "tic"]).sum())
    if duplicate_text_keys:
        raise ValueError(f"Text panel contains duplicate date/tic keys: {duplicate_text_keys}")
    duplicate_base_keys = int(base.duplicated(["date", "tic"]).sum())
    if duplicate_base_keys:
        raise ValueError(f"Base panel contains duplicate date/tic keys: {duplicate_base_keys}")

    merged = base.merge(text, on=["date", "tic"], how="left", validate="one_to_one", indicator=True)
    missing_text_rows = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns=["_merge"])
    for col in text_columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    output_raw_panel.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_raw_panel, index=False)
    return {
        "source_raw_panel": str(source_raw_panel),
        "text_panel": str(text_panel),
        "output_raw_panel": str(output_raw_panel),
        "rows": int(len(merged)),
        "columns_before": int(base.shape[1]),
        "columns_after": int(merged.shape[1]),
        "text_columns": text_columns,
        "missing_text_rows_zero_filled": missing_text_rows,
        "date_range": [str(merged["date"].min()), str(merged["date"].max())],
        "ticker_count": int(merged["tic"].nunique()),
        "text_non_zero_share": {
            col: float((merged[col].abs() > 1e-12).mean()) for col in text_columns
        },
    }


def make_text_config(
    *,
    source_config: Path,
    local_project: Path,
    output_config: Path,
    raw_panel_relative: str,
    text_columns: list[str],
) -> dict[str, Any]:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = "stage0_1_r6c_deepseek_v2_text"
    config["experiment"]["status"] = "ready_to_launch"
    config["experiment"]["purpose"] = (
        "R6c Stage0.1 baseline with DeepSeek v2 compact text features added "
        "as fixed PPO observation columns. Source RL repo remains read-only."
    )
    config["data"]["feature_set"] = "stage0_1_weight_features_plus_deepseek_v2_text10"
    config["data"]["raw_features_csv"] = raw_panel_relative
    config["data"]["model_ready_csv"] = (
        "artifacts/stage0_1_text/features/stage0_1_weight_features_model_ready_WITH_DEEPSEEK_V2_TEXT10.csv"
    )
    config["data"]["transform_stats_csv"] = (
        "artifacts/stage0_1_text/features/stage0_1_weight_feature_transform_stats_WITH_DEEPSEEK_V2_TEXT10.csv"
    )
    config["output"]["root_dir"] = "artifacts/stage0_1_text"
    config["output"]["run_name"] = "r6c_deepseek_v2_text_state_concat"
    config["output"]["skip_completed"] = True
    config.setdefault("instrumentation", {})
    config["instrumentation"]["save_sample_diagnostics"] = True
    config["instrumentation"]["save_rollout_snapshots"] = False
    config["instrumentation"]["rollout_snapshot_every_n_updates"] = 25
    config["text_features"] = {
        "source": "DeepSeek v2 compact PPO feature schema",
        "integration": "state_concat_via_stage0_1_weight_feature_panel",
        "normalization": "Stage0.1 fold_train_only scaling will be applied together with market features.",
        "columns": text_columns,
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "source_config": str(source_config),
        "output_config": str(output_config),
        "local_project": str(local_project),
        "variant": R6C_VARIANT,
        "text_columns": text_columns,
    }


def read_csv_from_zip(zp: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f)


def extract_baseline_zip_summaries(source_zip_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[pd.DataFrame] = []
    variant_summaries: list[pd.DataFrame] = []
    copied_members: list[dict[str, str]] = []

    for zp in sorted(source_zip_dir.glob("*.zip")):
        with zipfile.ZipFile(zp) as zf:
            members = zf.namelist()
            for member in members:
                keep = (
                    member.endswith("walk_forward_variant_summary.csv")
                    or member.endswith("walk_forward_validation_results.csv")
                    or member.endswith("/validation_summary.csv")
                    or member.endswith("/metadata.json")
                    or member.endswith("config.yaml")
                )
                if not keep:
                    continue
                target = output_dir / zp.stem / member
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied_members.append({"zip": str(zp), "member": member, "output": str(target)})

            variant_member = next((m for m in members if m.endswith("walk_forward_variant_summary.csv")), None)
            if variant_member:
                df = read_csv_from_zip(zp, variant_member)
                df["source_zip"] = zp.name
                variant_summaries.append(df)
            validation_member = next((m for m in members if m.endswith("walk_forward_validation_results.csv")), None)
            if validation_member:
                df = read_csv_from_zip(zp, validation_member)
                df["source_zip"] = zp.name
                fold_rows.append(df)

    all_validation = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    all_variants = pd.concat(variant_summaries, ignore_index=True) if variant_summaries else pd.DataFrame()
    if not all_validation.empty:
        all_validation.to_csv(output_dir / "r6c_validation_results_all_folds.csv", index=False)
    if not all_variants.empty:
        all_variants.to_csv(output_dir / "r6c_variant_summary_all_zips.csv", index=False)
    return {
        "source_zip_dir": str(source_zip_dir),
        "output_dir": str(output_dir),
        "zip_count": len(list(source_zip_dir.glob("*.zip"))),
        "copied_member_count": len(copied_members),
        "validation_rows": int(len(all_validation)),
        "variant_summary_rows": int(len(all_variants)),
        "copied_members": copied_members,
    }


def max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = values / np.maximum(running_max, 1e-12) - 1.0
    return float(np.min(drawdown))


def safe_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12 or not math.isfinite(std):
        return 0.0
    return float(np.sqrt(252.0) * np.mean(returns) / std)


def summarize_r6c_frozen(stage4_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "frozen_test_manifest.json",
        "frozen_test_behavior_log_daily.csv",
        "frozen_test_code_summary.csv",
        "test_codes.csv",
        "README_R6C_FROZEN_TEST_FOR_JOSEPH.md",
    ]:
        src = stage4_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    behavior = pd.read_csv(stage4_dir / "frozen_test_behavior_log_daily.csv")
    returns = pd.to_numeric(behavior["net_return"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    values = pd.to_numeric(behavior["portfolio_value"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    total_return = float(np.prod(1.0 + returns) - 1.0) if returns.size else 0.0
    summary = {
        "strategy": "R6c Stage0.1 baseline frozen test",
        "total_return": total_return,
        "sharpe": safe_sharpe(returns),
        "max_drawdown": max_drawdown(values),
        "rows": int(len(behavior)),
        "date_start": str(behavior["date"].min()),
        "date_end": str(behavior["date"].max()),
        "turnover_l1_mean": float(pd.to_numeric(behavior["turnover_l1"], errors="coerce").mean()),
        "source_file": str(stage4_dir / "frozen_test_behavior_log_daily.csv"),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "r6c_frozen_test_summary.csv", index=False)
    write_json(output_dir / "r6c_frozen_test_summary.json", summary)
    return summary


def build_comparison_report(
    *,
    out_root: Path,
    r6c_frozen_summary: dict[str, Any],
    validation_summary_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if OLD_BENCHMARK.exists():
        old = pd.read_csv(OLD_BENCHMARK)
        for _, row in old.iterrows():
            item = row.to_dict()
            item["comparison_scope"] = "old_finrl_frozen_test_2022_2023"
            item["source_file"] = str(OLD_BENCHMARK)
            rows.append(item)
    if OLD_TEXT_COMPARISON.exists():
        old_text = pd.read_csv(OLD_TEXT_COMPARISON)
        for _, row in old_text.iterrows():
            if str(row.get("strategy", "")) == "PPO with text":
                item = row.to_dict()
                item["comparison_scope"] = "old_finrl_text_ablation_2022_2023"
                rows.append(item)
    r6c_row = dict(r6c_frozen_summary)
    r6c_row["comparison_scope"] = "new_stage0_1_r6c_frozen_test_2022_2023"
    rows.append(r6c_row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_root / "comparison_summary_old_vs_r6c.csv", index=False)

    deltas: list[dict[str, Any]] = []
    old_ppo = comparison[comparison["strategy"].astype(str).eq("PPO without text")]
    if not old_ppo.empty:
        old_row = old_ppo.iloc[0]
        for metric in ["total_return", "sharpe", "max_drawdown"]:
            deltas.append(
                {
                    "metric": metric,
                    "old_ppo_without_text": float(old_row[metric]),
                    "new_r6c_frozen": float(r6c_frozen_summary[metric]),
                    "delta_new_minus_old": float(r6c_frozen_summary[metric]) - float(old_row[metric]),
                }
            )
    pd.DataFrame(deltas).to_csv(out_root / "comparison_deltas_old_vs_r6c.csv", index=False)

    report = out_root / "R6C_TEXT_LAUNCH_PREP.md"
    report.write_text(
        "\n".join(
            [
                "# R6c Stage0.1 Text Launch Preparation",
                "",
                "## Baseline Comparison",
                "",
                f"- Old PPO without text: total_return={float(old_ppo.iloc[0]['total_return']):.6f}, "
                f"sharpe={float(old_ppo.iloc[0]['sharpe']):.6f}, "
                f"max_drawdown={float(old_ppo.iloc[0]['max_drawdown']):.6f}."
                if not old_ppo.empty
                else "- Old PPO without text summary was not found.",
                f"- New R6c frozen baseline: total_return={r6c_frozen_summary['total_return']:.6f}, "
                f"sharpe={r6c_frozen_summary['sharpe']:.6f}, "
                f"max_drawdown={r6c_frozen_summary['max_drawdown']:.6f}.",
                "",
                "Interpretation: this is a baseline handoff comparison, not a final statistical claim. "
                "The old PPO and R6c Stage0.1 differ in action semantics and execution logic, but the "
                "frozen 2022-2023 window is aligned enough for launch triage.",
                "",
                "## Text Experiment",
                "",
                "- Integration: DeepSeek v2 compact text features are appended as fixed numeric state columns.",
                "- Normalization: Stage0.1 fold-train-only scaler will fit market and text columns together.",
                "- Source RL repo: read-only. All runnable files are copied into `rl_stage0_1_r6c_project/`.",
                "",
                "## Launch Commands",
                "",
                "Smoke:",
                "",
                "```powershell",
                'cd "artifacts\\r6c_stage0_1_text_baseline_20260530\\rl_stage0_1_r6c_project"',
                '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants '
                + R6C_VARIANT
                + " --folds fold_2021 --smoke-test --force",
                "```",
                "",
                "Full 4-fold screening:",
                "",
                "```powershell",
                'cd "artifacts\\r6c_stage0_1_text_baseline_20260530\\rl_stage0_1_r6c_project"',
                '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" -m src.ppo.stage0_1_train --config configs/stage0_1_r6c_deepseek_v2_text.yaml --variants '
                + R6C_VARIANT,
                "```",
                "",
                f"Validation summary extracted from new baseline zips: `{validation_summary_path}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "comparison_summary": str(out_root / "comparison_summary_old_vs_r6c.csv"),
        "comparison_deltas": str(out_root / "comparison_deltas_old_vs_r6c.csv"),
        "report": str(report),
    }


def write_launch_helpers(local_project: Path) -> None:
    smoke = local_project / "run_r6c_text_smoke.ps1"
    full = local_project / "run_r6c_text_full.ps1"
    command = (
        '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" '
        "-m src.ppo.stage0_1_train "
        "--config configs/stage0_1_r6c_deepseek_v2_text.yaml "
        f"--variants {R6C_VARIANT}"
    )
    smoke.write_text(command + " --folds fold_2021 --smoke-test --force\n", encoding="utf-8")
    full.write_text(command + "\n", encoding="utf-8")


def zip_launch_project(local_project: Path, zip_path: Path) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    excluded_parts = {"__pycache__", ".pytest_cache"}
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in local_project.rglob("*"):
            if path.is_dir():
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            rel = path.relative_to(local_project.parent)
            zf.write(path, rel.as_posix())
            file_count += 1
    return {"zip_path": str(zip_path), "file_count": file_count, "size_bytes": zip_path.stat().st_size}


def main() -> None:
    args = parse_args()
    source_rl_root = args.source_rl_root
    out_root = args.out_root
    local_project = out_root / "rl_stage0_1_r6c_project"
    source_zip_dir = source_rl_root / "artifacts" / "stage0_1" / R6C_VARIANT
    stage4_dir = source_rl_root / "artifacts" / "stage4" / (
        "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
    )
    source_raw_panel = source_rl_root / "artifacts" / "stage0_1" / "features" / "stage0_1_weight_features_raw.csv"

    text_columns = load_text_feature_columns(args.text_schema)
    copy_required_stage0_project(source_rl_root, local_project)
    local_features_dir = local_project / "artifacts" / "stage0_1" / "features"
    raw_with_text = local_features_dir / "stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv"
    merge_info = build_text_augmented_raw_panel(
        source_raw_panel=source_raw_panel,
        text_panel=args.text_panel,
        output_raw_panel=raw_with_text,
        text_columns=text_columns,
    )
    config_info = make_text_config(
        source_config=source_rl_root / "configs" / "stage0_1_active_r_pipeline.yaml",
        local_project=local_project,
        output_config=local_project / "configs" / "stage0_1_r6c_deepseek_v2_text.yaml",
        raw_panel_relative="artifacts/stage0_1/features/stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv",
        text_columns=text_columns,
    )
    write_launch_helpers(local_project)

    baseline_zip_info = extract_baseline_zip_summaries(source_zip_dir, out_root / "baseline_r6c_validation")
    r6c_frozen_summary = summarize_r6c_frozen(stage4_dir, out_root / "baseline_r6c_frozen_test")
    comparison_info = build_comparison_report(
        out_root=out_root,
        r6c_frozen_summary=r6c_frozen_summary,
        validation_summary_path=out_root / "baseline_r6c_validation" / "r6c_validation_results_all_folds.csv",
    )
    zip_info = None
    if not args.no_zip:
        zip_info = zip_launch_project(local_project, out_root / "r6c_stage0_1_deepseek_v2_text_launch_package.zip")

    manifest = {
        "status": "ready_to_launch",
        "created_by": "scripts/11_prepare_r6c_stage0_1_text_baseline.py",
        "source_rl_root_read_only": str(source_rl_root),
        "out_root": str(out_root),
        "local_project": str(local_project),
        "variant": R6C_VARIANT,
        "merge_info": merge_info,
        "config_info": config_info,
        "baseline_zip_info": baseline_zip_info,
        "r6c_frozen_summary": r6c_frozen_summary,
        "comparison_info": comparison_info,
        "zip_info": zip_info,
        "launch_commands": {
            "smoke": (
                '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" '
                "-m src.ppo.stage0_1_train "
                "--config configs/stage0_1_r6c_deepseek_v2_text.yaml "
                f"--variants {R6C_VARIANT} --folds fold_2021 --smoke-test --force"
            ),
            "full": (
                '& "C:\\Users\\ivanp\\anaconda3\\envs\\tensorflow\\python.exe" '
                "-m src.ppo.stage0_1_train "
                "--config configs/stage0_1_r6c_deepseek_v2_text.yaml "
                f"--variants {R6C_VARIANT}"
            ),
        },
    }
    write_json(out_root / "launch_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
