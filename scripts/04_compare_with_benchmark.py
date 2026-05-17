#!/usr/bin/env python
"""
Compare PPO with text against the frozen PPO-without-text benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = TOY_ROOT / "ppo_without_text_BENCHMARK" / "benchmark_summary.csv"
DEFAULT_TEXT_SUMMARY = TOY_ROOT / "artifacts" / "ppo_with_text_run" / "results" / "ppo_with_text_summary_for_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ppo-with-text-csv", type=Path, default=DEFAULT_TEXT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=TOY_ROOT / "artifacts" / "ppo_text_vs_benchmark")
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        help="Create a benchmark-only report if PPO-with-text results do not exist yet.",
    )
    return parser.parse_args()


def normalize_summary(path: Path, source_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    rename = {}
    if "return" in df.columns and "total_return" not in df.columns:
        rename["return"] = "total_return"
    if "sharpe_ratio" in df.columns and "sharpe" not in df.columns:
        rename["sharpe_ratio"] = "sharpe"
    if "max_drawdown_pct" in df.columns and "max_drawdown" not in df.columns:
        rename["max_drawdown_pct"] = "max_drawdown"
    df = df.rename(columns=rename)
    required = ["strategy", "total_return", "sharpe", "max_drawdown"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[required].copy()
    out["source_file"] = source_label
    return out


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(TOY_ROOT))
    except ValueError:
        return str(path)


def metric_delta(rows: pd.DataFrame) -> pd.DataFrame:
    strategies = rows["strategy"].astype(str).str.strip().str.lower()
    base = rows[strategies.eq("ppo without text")]
    if base.empty:
        base = rows[strategies.str.contains("ppo") & strategies.str.contains("without text")]
    text = rows[strategies.eq("ppo with text")]
    if text.empty:
        text = rows[strategies.str.contains("ppo") & strategies.str.contains("with text")]
    if base.empty or text.empty:
        return pd.DataFrame()
    b = base.iloc[0]
    t = text.iloc[0]
    return pd.DataFrame(
        [
            {
                "metric": "total_return",
                "ppo_without_text": b["total_return"],
                "ppo_with_text": t["total_return"],
                "delta": t["total_return"] - b["total_return"],
            },
            {
                "metric": "sharpe",
                "ppo_without_text": b["sharpe"],
                "ppo_with_text": t["sharpe"],
                "delta": t["sharpe"] - b["sharpe"],
            },
            {
                "metric": "max_drawdown",
                "ppo_without_text": b["max_drawdown"],
                "ppo_with_text": t["max_drawdown"],
                "delta": t["max_drawdown"] - b["max_drawdown"],
            },
        ]
    )


def write_svg(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("total_return", "Return"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max DD"),
    ]
    width, height = 760, 320
    margin_left, top = 90, 40
    group_w = 210
    bar_w = 34
    colors = ["#d88b38", "#7fb069", "#b85c5c", "#6c8ebf"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff8ef"/>',
        '<text x="24" y="26" font-family="Arial" font-size="18" font-weight="700" fill="#2f261f">PPO Text Ablation Comparison</text>',
    ]
    for gi, (metric, label) in enumerate(metrics):
        x0 = margin_left + gi * group_w
        vals = [float(v) for v in rows[metric].fillna(0).tolist()]
        scale = max(max(abs(v) for v in vals), 0.05)
        baseline_y = top + 120
        svg.append(f'<line x1="{x0}" y1="{baseline_y}" x2="{x0 + 150}" y2="{baseline_y}" stroke="#d9c7b8"/>')
        svg.append(
            f'<text x="{x0}" y="{height - 34}" font-family="Arial" font-size="13" font-weight="700" fill="#2f261f">{xml_escape(label)}</text>'
        )
        for i, (_, row) in enumerate(rows.iterrows()):
            value = float(row[metric])
            bar_h = min(105, abs(value) / scale * 105)
            x = x0 + 16 + i * (bar_w + 10)
            y = baseline_y - bar_h if value >= 0 else baseline_y
            svg.append(
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{colors[i % len(colors)]}"/>'
            )
            svg.append(
                f'<text x="{x}" y="{baseline_y + bar_h + 16 if value < 0 else y - 6}" font-family="Arial" font-size="10" fill="#2f261f">{value:.3f}</text>'
            )
    legend_y = height - 10
    for i, strategy in enumerate(rows["strategy"].tolist()):
        x = 24 + i * 180
        svg.append(f'<rect x="{x}" y="{legend_y - 11}" width="10" height="10" fill="{colors[i % len(colors)]}"/>')
        svg.append(
            f'<text x="{x + 16}" y="{legend_y - 2}" font-family="Arial" font-size="11" fill="#2f261f">{xml_escape(str(strategy))}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    cols = [str(col) for col in df.columns]
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(rows: pd.DataFrame, deltas: pd.DataFrame, path: Path, status: str) -> None:
    lines = [
        "# PPO Text Ablation Report",
        "",
        f"Status: `{status}`",
        "",
        "## Summary",
        "",
        df_to_markdown(rows),
        "",
    ]
    if not deltas.empty:
        lines += ["## Delta: PPO with text minus PPO without text", "", df_to_markdown(deltas), ""]
    else:
        lines += [
            "## Delta",
            "",
            "PPO-with-text result is not available yet, so only the frozen benchmark is shown.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.output_dir / "results"
    figures_dir = args.output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = [normalize_summary(args.benchmark_csv, portable_path(args.benchmark_csv))]
    status = "completed"
    if args.ppo_with_text_csv.exists():
        frames.append(normalize_summary(args.ppo_with_text_csv, portable_path(args.ppo_with_text_csv)))
    elif args.allow_missing_text:
        status = "benchmark_only_missing_text_result"
    else:
        raise FileNotFoundError(
            f"PPO-with-text summary not found: {args.ppo_with_text_csv}. "
            "Run scripts/03_train_backtest_ppo_with_text.py first, or pass --allow-missing-text."
        )

    rows = pd.concat(frames, ignore_index=True)
    deltas = metric_delta(rows)
    comparison_path = results_dir / "comparison_summary.csv"
    delta_path = results_dir / "comparison_deltas.csv"
    report_path = args.output_dir / "report.md"
    figure_path = figures_dir / "return_sharpe_drawdown.svg"
    manifest_path = args.output_dir / "manifest.json"

    rows.to_csv(comparison_path, index=False)
    deltas.to_csv(delta_path, index=False)
    write_svg(rows, figure_path)
    write_report(rows, deltas, report_path, status)
    manifest = {
        "status": status,
        "benchmark_csv": portable_path(args.benchmark_csv),
        "ppo_with_text_csv": portable_path(args.ppo_with_text_csv),
        "comparison_summary": portable_path(comparison_path),
        "comparison_deltas": portable_path(delta_path),
        "report": portable_path(report_path),
        "figure": portable_path(figure_path),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
