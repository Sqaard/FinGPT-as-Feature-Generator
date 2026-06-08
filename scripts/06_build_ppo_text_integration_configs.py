#!/usr/bin/env python
"""Build PPO-side text integration experiment configs and risk-overlay panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PANEL = TOY_ROOT / "artifacts" / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv"
DEFAULT_NORMALIZED_PANEL = (
    TOY_ROOT
    / "artifacts"
    / "normalized_text_panels"
    / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_robust_clipped.csv"
)
DEFAULT_OUTPUT_DIR = TOY_ROOT / "artifacts" / "ppo_text_integration_configs"
DEFAULT_REPORT = TOY_ROOT / "reports" / "drl_ppo_text_integration.md"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-panel", type=Path, default=DEFAULT_RAW_PANEL)
    parser.add_argument("--normalized-panel", type=Path, default=DEFAULT_NORMALIZED_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-start", default="2010-01-04")
    parser.add_argument("--train-end", default="2021-09-30")
    parser.add_argument("--risk-overlay-weight", type=float, default=0.5)
    parser.add_argument("--timesteps", type=int, default=350_000)
    return parser.parse_args()


def compute_text_risk_score(df: pd.DataFrame) -> pd.Series:
    risk = pd.to_numeric(df.get("text_risk_intensity", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    uncertainty = pd.to_numeric(df.get("text_uncertainty_intensity", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    event_risk = pd.to_numeric(df.get("text_company_event_risk_impact", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    macro_risk = (-pd.to_numeric(df.get("text_macro_financial_conditions_impact", 0.0), errors="coerce").fillna(0.0)).clip(lower=0.0)
    opportunity = pd.to_numeric(df.get("text_opportunity_intensity", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    score = (0.35 * risk) + (0.25 * uncertainty) + (0.25 * event_risk) + (0.15 * macro_risk) - (0.20 * opportunity)
    return score.clip(lower=0.0)


def build_risk_overlay_panel(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    df = pd.read_csv(args.normalized_panel)
    missing = [col for col in TEXT_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Normalized panel is missing text columns: {missing}")
    if "turbulence" not in df.columns:
        raise ValueError("Panel must contain turbulence for risk-overlay experiments.")

    dates = pd.to_datetime(df["date"])
    train_mask = (dates >= pd.Timestamp(args.train_start)) & (dates <= pd.Timestamp(args.train_end))
    train_turbulence = pd.to_numeric(df.loc[train_mask, "turbulence"], errors="coerce").fillna(0.0)
    base_scale = float(train_turbulence.quantile(0.95))
    if base_scale <= 0:
        base_scale = 100.0

    risk_score = compute_text_risk_score(df)
    df["text_risk_control_score"] = risk_score
    df["turbulence_text_overlay"] = (
        pd.to_numeric(df["turbulence"], errors="coerce").fillna(0.0)
        + args.risk_overlay_weight * base_scale * risk_score
    )

    output_path = args.output_dir / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL_text_risk_overlay.csv"
    df.to_csv(output_path, index=False)
    summary = {
        "output": portable_path(output_path),
        "train_turbulence_p95": base_scale,
        "risk_overlay_weight": args.risk_overlay_weight,
        "text_risk_score_mean": float(risk_score.mean()),
        "text_risk_score_p95": float(risk_score.quantile(0.95)),
        "overlay_turbulence_p95": float(df["turbulence_text_overlay"].quantile(0.95)),
    }
    return output_path, summary


def command(panel: Path, output_dir: str, strategy: str, timesteps: int, extra: str = "") -> str:
    suffix = f" {extra}" if extra else ""
    panel_arg = portable_path(panel)
    return (
        "python scripts/03_train_backtest_ppo_with_text.py "
        f'--panel "{panel_arg}" '
        f"--output-dir artifacts/{output_dir} "
        f"--timesteps {timesteps} "
        f"--text-integration-strategy {strategy}{suffix}"
    )


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TOY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["variant", "strategy", "panel", "purpose"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], overlay_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DRL Issue #6: PPO-side text integration strategies",
        "",
        "Goal: test whether text works better outside raw feature concatenation.",
        "",
        "## Experiment matrix",
        "",
        markdown_table(rows),
        "",
        "## Risk overlay",
        "",
        "The risk-overlay variant converts text risk, uncertainty, macro stress, and event risk into a `turbulence_text_overlay` column.",
        "This lets the existing FinRL risk-control path react to text without changing the environment internals.",
        "",
        "```json",
        json.dumps(overlay_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Two-branch policy prototype",
        "",
        "`scripts/03_train_backtest_ppo_with_text.py --text-integration-strategy two_branch_policy` now builds a custom SB3 feature extractor.",
        "It sends market features and text features through separate MLP branches before PPO receives the merged representation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    overlay_panel, overlay_summary = build_risk_overlay_panel(args)
    rows = [
        {
            "variant": "state_concat_raw",
            "strategy": "state_concat",
            "panel": portable_path(args.raw_panel),
            "purpose": "current baseline: raw text directly in PPO state",
            "command": command(args.raw_panel, "ppo_state_concat_raw", "state_concat", args.timesteps),
        },
        {
            "variant": "state_concat_robust_clipped",
            "strategy": "state_concat",
            "panel": portable_path(args.normalized_panel),
            "purpose": "same architecture, train-only robust clipped text",
            "command": command(args.normalized_panel, "ppo_state_concat_robust_clipped", "state_concat", args.timesteps),
        },
        {
            "variant": "market_only_control",
            "strategy": "market_only",
            "panel": portable_path(args.normalized_panel),
            "purpose": "control run: panel contains text but PPO state excludes text columns",
            "command": command(args.normalized_panel, "ppo_market_only_control", "market_only", args.timesteps),
        },
        {
            "variant": "text_risk_overlay",
            "strategy": "text_risk_overlay",
            "panel": portable_path(overlay_panel),
            "purpose": "text affects risk control through turbulence overlay",
            "command": command(
                overlay_panel,
                "ppo_text_risk_overlay",
                "text_risk_overlay",
                args.timesteps,
                "--risk-indicator-col turbulence_text_overlay --turbulence-threshold 100",
            ),
        },
        {
            "variant": "two_branch_policy",
            "strategy": "two_branch_policy",
            "panel": portable_path(args.normalized_panel),
            "purpose": "market/text separate encoders before PPO policy/value heads",
            "command": command(args.normalized_panel, "ppo_two_branch_policy", "two_branch_policy", args.timesteps),
        },
    ]

    matrix_path = args.output_dir / "experiment_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    manifest = {
        "status": "completed",
        "experiment_matrix": portable_path(matrix_path),
        "report": portable_path(args.report),
        "overlay_panel": portable_path(overlay_panel),
        "overlay_summary": overlay_summary,
        "variants": rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.report, rows, overlay_summary)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
