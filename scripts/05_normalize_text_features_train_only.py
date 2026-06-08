#!/usr/bin/env python
"""Create PPO panels with train-only text-feature normalization.

The key leakage rule is simple: all scaler parameters are fitted only on the
training window, then reused unchanged for validation and test rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = TOY_ROOT / "artifacts" / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv"
DEFAULT_OUTPUT_DIR = TOY_ROOT / "artifacts" / "normalized_text_panels"
DEFAULT_REPORT = TOY_ROOT / "reports" / "drl_train_only_normalization.md"

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

DEFAULT_METHODS = ["zscore", "robust", "clipped", "zscore_clipped", "robust_clipped"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--text-feature-schema",
        type=Path,
        default=None,
        help="Optional schema JSON. When provided, normalize these feature names instead of the v1 defaults.",
    )
    parser.add_argument(
        "--text-feature-columns",
        nargs="*",
        default=None,
        help="Optional explicit text feature columns. Overrides --text-feature-schema.",
    )
    parser.add_argument("--train-start", default="2010-01-04")
    parser.add_argument("--train-end", default="2021-09-30")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--clip-value", type=float, default=3.0)
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.01,
        help="Two-sided train-window quantile for raw clipped panels.",
    )
    return parser.parse_args()


def _out_col(feature: str) -> str:
    return feature if feature.startswith("text_") else f"text_{feature}"


def load_text_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.text_feature_columns:
        return [_out_col(item) for item in args.text_feature_columns]
    if args.text_feature_schema:
        payload = json.loads(args.text_feature_schema.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"Schema has no features: {args.text_feature_schema}")
        return [_out_col(str(item["name"])) for item in features]
    return TEXT_FEATURE_COLUMNS


def safe_scale(value: float, fallback: float = 1.0) -> float:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return fallback
    return float(value)


def fit_train_stats(
    df: pd.DataFrame,
    train_mask: pd.Series,
    clip_quantile: float,
    text_feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in text_feature_columns:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        train_values = values.loc[train_mask]
        q_low = float(train_values.quantile(clip_quantile))
        q_high = float(train_values.quantile(1.0 - clip_quantile))
        iqr = float(train_values.quantile(0.75) - train_values.quantile(0.25))
        stats[col] = {
            "mean": float(train_values.mean()),
            "std": safe_scale(float(train_values.std(ddof=0))),
            "median": float(train_values.median()),
            "iqr": safe_scale(iqr),
            "clip_low": q_low,
            "clip_high": q_high if q_high > q_low else q_low + 1e-12,
            "train_min": float(train_values.min()),
            "train_max": float(train_values.max()),
        }
    return stats


def transform_column(values: pd.Series, stat: dict[str, float], method: str, clip_value: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if method == "zscore":
        return (numeric - stat["mean"]) / stat["std"]
    if method == "robust":
        return (numeric - stat["median"]) / stat["iqr"]
    if method == "clipped":
        return numeric.clip(stat["clip_low"], stat["clip_high"])
    if method == "zscore_clipped":
        return ((numeric - stat["mean"]) / stat["std"]).clip(-clip_value, clip_value)
    if method == "robust_clipped":
        return ((numeric - stat["median"]) / stat["iqr"]).clip(-clip_value, clip_value)
    raise ValueError(f"Unknown method: {method}")


def describe_panel(df: pd.DataFrame, train_mask: pd.Series, text_feature_columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, mask in {
        "train": train_mask,
        "validation_test": ~train_mask,
    }.items():
        frame = df.loc[mask, text_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out[split_name] = {
            "rows": int(mask.sum()),
            "mean_abs_feature_mean": float(frame.mean().abs().mean()),
            "max_abs_value": float(frame.abs().max().max()),
            "non_zero_share": float((frame.abs() > 1e-12).mean().mean()),
        }
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, source_panel: Path, manifest_rows: list[dict[str, Any]], stats_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DRL Issue #5: Train-only text normalization",
        "",
        "Goal: make text features safer for PPO without leaking validation or test information.",
        "",
        f"Source panel: `{portable_path(source_panel)}`",
        f"Scaler statistics: `{portable_path(stats_path)}`",
        "",
        "## Leakage rule",
        "",
        "All means, standard deviations, medians, IQRs, and clipping bounds are fitted on the train window only.",
        "The same fitted parameters are then applied to every later row.",
        "",
        "## Generated panels",
        "",
        markdown_table(
            manifest_rows,
            [
                "method",
                "output",
                "train_rows",
                "validation_test_rows",
                "train_max_abs_value",
                "validation_test_max_abs_value",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- `zscore`: useful when PPO benefits from centered Gaussian-like features.",
        "- `robust`: safer when filings create heavy-tailed one-day spikes.",
        "- `clipped`: tests whether raw features were mostly fine but outliers hurt PPO.",
        "- clipped scaled variants are the first candidates for a stable DRL run.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TOY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.panel)
    text_feature_columns = load_text_feature_columns(args)
    missing = [col for col in text_feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing text columns: {missing}")
    if "date" not in df.columns:
        raise ValueError("Panel must contain a date column.")

    dates = pd.to_datetime(df["date"])
    train_mask = (dates >= pd.Timestamp(args.train_start)) & (dates <= pd.Timestamp(args.train_end))
    if not train_mask.any():
        raise ValueError("Train window has no rows; check --train-start and --train-end.")

    stats = fit_train_stats(df, train_mask, args.clip_quantile, text_feature_columns)
    stats_payload = {
        "source_panel": portable_path(args.panel),
        "train_window": [args.train_start, args.train_end],
        "fit_rows": int(train_mask.sum()),
        "validation_test_rows": int((~train_mask).sum()),
        "text_feature_columns": text_feature_columns,
        "stats": stats,
        "leakage_policy": "fit_on_train_window_only",
    }
    stats_path = args.output_dir / "train_only_scaler_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_rows: list[dict[str, Any]] = []
    for method in args.methods:
        out = df.copy()
        for col in text_feature_columns:
            out[col] = transform_column(out[col], stats[col], method, args.clip_value).astype(float)
        output_path = args.output_dir / f"{args.panel.stem}_{method}.csv"
        out.to_csv(output_path, index=False)
        summary = describe_panel(out, train_mask, text_feature_columns)
        manifest_rows.append(
            {
                "method": method,
            "output": portable_path(output_path),
                "train_rows": summary["train"]["rows"],
                "validation_test_rows": summary["validation_test"]["rows"],
                "train_max_abs_value": summary["train"]["max_abs_value"],
                "validation_test_max_abs_value": summary["validation_test"]["max_abs_value"],
                "train_non_zero_share": summary["train"]["non_zero_share"],
                "validation_test_non_zero_share": summary["validation_test"]["non_zero_share"],
            }
        )

    manifest = {
        "status": "completed",
        "source_panel": portable_path(args.panel),
        "output_dir": portable_path(args.output_dir),
        "report": portable_path(args.report),
        "scaler_stats": portable_path(stats_path),
        "methods": args.methods,
        "generated_panels": manifest_rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.report, args.panel, manifest_rows, stats_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
