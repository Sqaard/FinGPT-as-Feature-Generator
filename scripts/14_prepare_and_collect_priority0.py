#!/usr/bin/env python
"""Prepare and collect the Priority 0 R6c text experiments from plan.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
RAW_TEXT_PANEL = (
    R6C_ROOT
    / "artifacts"
    / "stage0_1"
    / "features"
    / "stage0_1_weight_features_raw_WITH_DEEPSEEK_V2_TEXT10.csv"
)
EXPERIMENT_ROOT = R6C_ROOT / "artifacts" / "text_improvement_plan"
REPO_RESULTS_ROOT = REPO_ROOT / "artifacts" / "text_improvement_plan"

R6C_VARIANT = "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"
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
FORWARD_EARNINGS_COLUMNS = [
    "text_earnings_pressure",
    "text_numeric_evidence_density",
    "text_signal_confidence",
]
VARIANTS = [
    "r6c_base",
    "r6c_text10_raw",
    "r6c_forward_earnings_raw",
    "r6c_forward_earnings_zero",
    "r6c_forward_earnings_shuffled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--r6c-root", type=Path, default=R6C_ROOT)
    prepare.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    prepare.add_argument("--raw-panel", type=Path, default=RAW_TEXT_PANEL)
    prepare.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    prepare.add_argument("--folds", nargs="+", default=["fold_2018", "fold_2019", "fold_2020", "fold_2021"])
    prepare.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    prepare.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    prepare.add_argument("--force-panels", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--r6c-root", type=Path, default=R6C_ROOT)
    collect.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    collect.add_argument("--repo-results-root", type=Path, default=REPO_RESULTS_ROOT)

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def relative_to_r6c(path: Path, r6c_root: Path) -> str:
    return str(path.resolve().relative_to(r6c_root.resolve()))


def fold_table(config: dict, r6c_root: Path) -> pd.DataFrame:
    path = Path(config["walk_forward"]["folds_csv"])
    if not path.is_absolute():
        path = r6c_root / path
    return pd.read_csv(path)


def materialize_panel(
    source: pd.DataFrame,
    *,
    variant: str,
    fold: pd.Series,
    path: Path,
    force: bool,
) -> None:
    if path.exists() and not force:
        return

    if variant == "r6c_base":
        panel = source.drop(columns=TEXT_COLUMNS)
    else:
        panel = source.copy()
        inactive = [column for column in TEXT_COLUMNS if column not in FORWARD_EARNINGS_COLUMNS]
        if variant in {
            "r6c_forward_earnings_raw",
            "r6c_forward_earnings_zero",
            "r6c_forward_earnings_shuffled",
        }:
            panel.loc[:, inactive] = 0.0
        if variant == "r6c_forward_earnings_zero":
            panel.loc[:, FORWARD_EARNINGS_COLUMNS] = 0.0
        elif variant == "r6c_forward_earnings_shuffled":
            train_start = pd.Timestamp(fold["train_start"])
            train_end = pd.Timestamp(fold["train_end_inclusive"])
            dates = pd.to_datetime(panel["date"])
            train_mask = dates.between(train_start, train_end)
            train_dates = pd.Index(sorted(dates.loc[train_mask].unique()))
            shuffled_dates = train_dates.to_series().sample(frac=1.0, random_state=9173).to_numpy()
            date_map = dict(zip(train_dates, shuffled_dates))
            source_by_date = (
                panel.loc[train_mask, ["date", "tic", *FORWARD_EARNINGS_COLUMNS]]
                .assign(date=pd.to_datetime(panel.loc[train_mask, "date"]))
                .set_index(["date", "tic"])
            )
            target = panel.loc[train_mask, ["date", "tic"]].copy()
            target["date"] = pd.to_datetime(target["date"]).map(date_map)
            shuffled = source_by_date.reindex(pd.MultiIndex.from_frame(target)).reset_index(drop=True)
            panel.loc[train_mask, FORWARD_EARNINGS_COLUMNS] = shuffled[FORWARD_EARNINGS_COLUMNS].to_numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)


def prepare(args: argparse.Namespace) -> None:
    r6c_root = args.r6c_root.resolve()
    experiment_root = args.experiment_root.resolve()
    config = load_yaml(args.base_config.resolve())
    folds = fold_table(config, r6c_root)
    folds = folds[folds["fold"].isin(args.folds)].copy()
    missing_folds = sorted(set(args.folds).difference(folds["fold"]))
    if missing_folds:
        raise ValueError(f"Unknown folds: {missing_folds}")

    header = pd.read_csv(args.raw_panel, nrows=0).columns
    missing_text = [column for column in TEXT_COLUMNS if column not in header]
    if missing_text:
        raise ValueError(f"Raw panel is missing text columns: {missing_text}")

    source: pd.DataFrame | None = None
    rows: list[dict] = []
    for variant in args.variants:
        for _, fold in folds.iterrows():
            fold_id = str(fold["fold"])
            if variant == "r6c_text10_raw":
                panel_path = args.raw_panel.resolve()
            else:
                panel_path = experiment_root / "panels" / variant / f"{fold_id}.csv"
                if source is None:
                    source = pd.read_csv(args.raw_panel)
                materialize_panel(
                    source,
                    variant=variant,
                    fold=fold,
                    path=panel_path,
                    force=args.force_panels,
                )

            for seed in args.seeds:
                run_name = f"{variant}/seed_{seed}/{fold_id}"
                seed_config = load_yaml(args.base_config.resolve())
                seed_config["experiment"]["name"] = f"text_improvement_{variant}_seed_{seed}"
                seed_config["data"]["feature_set"] = variant
                seed_config["data"]["raw_features_csv"] = relative_to_r6c(panel_path, r6c_root)
                seed_config["output"]["root_dir"] = relative_to_r6c(experiment_root / "runs", r6c_root)
                seed_config["output"]["run_name"] = run_name
                seed_config["output"]["skip_completed"] = True
                seed_config["walk_forward"]["fold_ids"] = [fold_id]
                seed_config["ppo"]["seed"] = int(seed)
                seed_config["text_features"] = {
                    "experiment_variant": variant,
                    "active_columns": (
                        []
                        if variant == "r6c_base"
                        else TEXT_COLUMNS
                        if variant == "r6c_text10_raw"
                        else FORWARD_EARNINGS_COLUMNS
                    ),
                    "control": (
                        "zero"
                        if variant == "r6c_forward_earnings_zero"
                        else "train_date_shuffle"
                        if variant == "r6c_forward_earnings_shuffled"
                        else "none"
                    ),
                }
                config_path = experiment_root / "configs" / variant / f"seed_{seed}_{fold_id}.yaml"
                write_yaml(config_path, seed_config)
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "fold": fold_id,
                        "config": relative_to_r6c(config_path, r6c_root),
                        "run_name": run_name,
                        "panel": relative_to_r6c(panel_path, r6c_root),
                        "r6c_variant": R6C_VARIANT,
                    }
                )

    manifest = pd.DataFrame(rows).sort_values(["variant", "seed", "fold"])
    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(experiment_root / "priority0_manifest.csv", index=False)
    (experiment_root / "priority0_manifest.json").write_text(
        json.dumps(manifest.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest)} jobs in {experiment_root}")


def collect(args: argparse.Namespace) -> None:
    r6c_root = args.r6c_root.resolve()
    experiment_root = args.experiment_root.resolve()
    manifest_path = experiment_root / "priority0_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare first: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    rows: list[dict] = []
    for job in manifest.to_dict(orient="records"):
        summary_path = (
            experiment_root
            / "runs"
            / str(job["run_name"])
            / R6C_VARIANT
            / str(job["fold"])
            / "validation_summary.csv"
        )
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        rows.append(
            {
                "variant": job["variant"],
                "seed": int(job["seed"]),
                "fold": job["fold"],
                "return": float(summary["return_pct"]),
                "sharpe": float(summary["sharpe"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "turnover_l1_mean": float(summary["turnover_l1_mean"]),
                "summary_path": str(summary_path.relative_to(r6c_root)),
            }
        )

    output_root = args.repo_results_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(output_root / "text_experiment_runs.csv", index=False)
    if runs.empty:
        print("No completed validation summaries found.")
        return

    baseline = runs[runs["variant"].eq("r6c_base")][
        ["seed", "fold", "return", "sharpe", "max_drawdown", "turnover_l1_mean"]
    ].rename(
        columns={
            "return": "baseline_return",
            "sharpe": "baseline_sharpe",
            "max_drawdown": "baseline_max_drawdown",
            "turnover_l1_mean": "baseline_turnover_l1_mean",
        }
    )
    paired = runs.merge(baseline, on=["seed", "fold"], how="left")
    for metric in ["return", "sharpe", "max_drawdown", "turnover_l1_mean"]:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"baseline_{metric}"]
    paired.to_csv(output_root / "text_experiment_paired.csv", index=False)

    summary_rows = []
    for variant, group in paired.groupby("variant", sort=False):
        valid = group.dropna(subset=["sharpe_delta"])
        summary_rows.append(
            {
                "variant": variant,
                "completed_runs": len(group),
                "paired_runs": len(valid),
                "mean_return": group["return"].mean(),
                "std_return": group["return"].std(),
                "mean_sharpe": group["sharpe"].mean(),
                "std_sharpe": group["sharpe"].std(),
                "worst_sharpe": group["sharpe"].min(),
                "mean_max_drawdown": group["max_drawdown"].mean(),
                "mean_turnover_l1": group["turnover_l1_mean"].mean(),
                "mean_return_delta": valid["return_delta"].mean(),
                "mean_sharpe_delta": valid["sharpe_delta"].mean(),
                "positive_sharpe_delta_fraction": (valid["sharpe_delta"] > 0).mean() if len(valid) else float("nan"),
                "mean_max_drawdown_delta": valid["max_drawdown_delta"].mean(),
                "mean_turnover_delta": valid["turnover_l1_mean_delta"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "text_experiment_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
