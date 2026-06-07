#!/usr/bin/env python
"""Build a cross-model table for text-feature impact.

The table compares each architecture against its own text-augmented version:

- old custom_custom PPO vs custom_custom with text10
- R6c Stage0.1 frozen baseline vs R6c with text10

It intentionally reports within-family deltas. The two model families are not
treated as identical policies, but the direction and magnitude of the text
effect can be compared.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "r6c_stage0_1_text_baseline_20260530"

CUSTOM_BASE_SUMMARY = REPO_ROOT / "ppo_without_text_BENCHMARK" / "benchmark_summary.csv"
CUSTOM_BASE_MANIFEST = REPO_ROOT / "ppo_without_text_BENCHMARK" / "results" / "export_manifest.json"
CUSTOM_TEXT_SUMMARY = REPO_ROOT / "artifacts" / "ppo_with_text_run" / "results" / "ppo_with_text_test_summary.csv"
R6C_BASE_SUMMARY = DEFAULT_OUT_DIR / "comparison_summary_old_vs_r6c.csv"
R6C_TEXT_OOS = (
    DEFAULT_OUT_DIR
    / "rl_stage0_1_r6c_project"
    / "artifacts"
    / "stage0_1_text"
    / "r6c_deepseek_v2_text_frozen_oos"
    / "frozen_oos_results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if str(row.get(key, "")) == value]
    if not matches:
        raise ValueError(f"No row with {key}={value!r}")
    return matches[0]


def custom_period(manifest: dict[str, Any], name: str) -> tuple[str, str]:
    periods = manifest.get("periods", {})
    period = periods.get(name, ["", ""])
    return str(period[0]), str(period[1])


def build_rows() -> list[dict[str, Any]]:
    custom_manifest = read_json(CUSTOM_BASE_MANIFEST)
    custom_test_start, custom_test_end = custom_period(custom_manifest, "test")

    custom_base = find_row(read_csv_rows(CUSTOM_BASE_SUMMARY), "strategy", "PPO without text")
    custom_text = find_row(read_csv_rows(CUSTOM_TEXT_SUMMARY), "strategy", "PPO with text")

    r6c_base_rows = read_csv_rows(R6C_BASE_SUMMARY)
    r6c_base = find_row(r6c_base_rows, "strategy", "R6c Stage0.1 baseline frozen test")
    r6c_text = find_row(read_csv_rows(R6C_TEXT_OOS), "fold", "fold_2021")

    r6c_start = r6c_base.get("date_start", "2022-01-03")
    r6c_end = r6c_base.get("date_end", "2023-02-27")

    return [
        {
            "model_label": "custom_custom",
            "architecture_family": "custom_custom",
            "text_setting": "none",
            "text_feature_count": 0,
            "period": "test",
            "date_start": custom_test_start,
            "date_end": custom_test_end,
            "days": "",
            "total_return": fnum(custom_base["total_return"]),
            "sharpe": fnum(custom_base["sharpe"]),
            "max_drawdown": fnum(custom_base["max_drawdown"]),
            "turnover_l1_mean": "",
            "source_file": rel(CUSTOM_BASE_SUMMARY),
            "notes": "Old selected custom_custom no-text benchmark; summary is rounded in source CSV.",
        },
        {
            "model_label": "custom_custom&text10",
            "architecture_family": "custom_custom",
            "text_setting": "mistral_text_10_state_concat",
            "text_feature_count": int(float(custom_text.get("text_feature_count", 10))),
            "period": "test",
            "date_start": custom_test_start,
            "date_end": custom_test_end,
            "days": "",
            "total_return": fnum(custom_text["cumulative_returns"]),
            "sharpe": fnum(custom_text["sharpe_ratio"]),
            "max_drawdown": fnum(custom_text["max_drawdown"]),
            "turnover_l1_mean": "",
            "source_file": rel(CUSTOM_TEXT_SUMMARY),
            "notes": "Old custom_custom PPO retrained with 10 text columns.",
        },
        {
            "model_label": "R6c",
            "architecture_family": "R6c",
            "text_setting": "none",
            "text_feature_count": 0,
            "period": "frozen_oos_fold_2021",
            "date_start": r6c_start,
            "date_end": r6c_end,
            "days": int(float(r6c_base.get("rows", 0))) if r6c_base.get("rows") else "",
            "total_return": fnum(r6c_base["total_return"]),
            "sharpe": fnum(r6c_base["sharpe"]),
            "max_drawdown": fnum(r6c_base["max_drawdown"]),
            "turnover_l1_mean": fnum(r6c_base["turnover_l1_mean"]),
            "source_file": r6c_base["source_file"],
            "notes": "Frozen R6c Stage0.1 baseline from external RL project copy.",
        },
        {
            "model_label": "R6c&text10",
            "architecture_family": "R6c",
            "text_setting": "deepseek_v2_text10_state_concat",
            "text_feature_count": 10,
            "period": "frozen_oos_fold_2021",
            "date_start": r6c_start,
            "date_end": r6c_end,
            "days": int(float(r6c_text["days"])),
            "total_return": fnum(r6c_text["return_pct"]),
            "sharpe": fnum(r6c_text["sharpe"]),
            "max_drawdown": fnum(r6c_text["max_drawdown"]),
            "turnover_l1_mean": fnum(r6c_text["turnover_l1_mean"]),
            "source_file": rel(R6C_TEXT_OOS),
            "notes": "Frozen OOS evaluation of trained R6c text10 policy on fold_2021.",
        },
    ]


def build_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {str(row["model_label"]): row for row in rows}
    pairs = [
        ("custom_custom", "custom_custom&text10"),
        ("R6c", "R6c&text10"),
    ]
    deltas: list[dict[str, Any]] = []
    for base_label, text_label in pairs:
        base = by_label[base_label]
        text = by_label[text_label]
        delta_return = float(text["total_return"]) - float(base["total_return"])
        delta_sharpe = float(text["sharpe"]) - float(base["sharpe"])
        delta_dd = float(text["max_drawdown"]) - float(base["max_drawdown"])
        deltas.append(
            {
                "architecture_family": base["architecture_family"],
                "base_model": base_label,
                "text_model": text_label,
                "base_total_return": base["total_return"],
                "text_total_return": text["total_return"],
                "delta_total_return": delta_return,
                "base_sharpe": base["sharpe"],
                "text_sharpe": text["sharpe"],
                "delta_sharpe": delta_sharpe,
                "base_max_drawdown": base["max_drawdown"],
                "text_max_drawdown": text["max_drawdown"],
                "delta_max_drawdown": delta_dd,
                "effect_direction": "better" if delta_return > 0 and delta_sharpe > 0 else "worse",
                "notes": "For max_drawdown, a more negative delta means drawdown worsened.",
            }
        )
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def num(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


def write_report(path: Path, rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Model Text Effect Comparison",
        "",
        "This table compares text impact within each PPO architecture family. It should not be read as a clean head-to-head architecture benchmark, because `custom_custom` and `R6c` have different action/execution mechanics.",
        "",
        "## Model Rows",
        "",
        "| model | period | return | Sharpe | max DD | turnover | source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {period} | {ret} | {sharpe} | {dd} | {turnover} | `{source}` |".format(
                model=row["model_label"],
                period=row["period"],
                ret=pct(row["total_return"]),
                sharpe=num(row["sharpe"]),
                dd=pct(row["max_drawdown"]),
                turnover=num(row["turnover_l1_mean"]),
                source=row["source_file"],
            )
        )

    lines.extend(
        [
            "",
            "## Text Deltas",
            "",
            "| family | return delta | Sharpe delta | max DD delta | direction |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in deltas:
        lines.append(
            "| {family} | {ret} | {sharpe} | {dd} | {direction} |".format(
                family=row["architecture_family"],
                ret=pct(row["delta_total_return"]),
                sharpe=num(row["delta_sharpe"]),
                dd=pct(row["delta_max_drawdown"]),
                direction=row["effect_direction"],
            )
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- On `custom_custom`, text10 made the test result materially worse: return and Sharpe both fell strongly.",
            "- On `R6c`, text10 slightly improved return and Sharpe, while max drawdown became slightly worse.",
            "- Current evidence says text features interact better with the R6c hierarchy than with the old flat `custom_custom` policy, but the R6c gain is still small and should be treated as screening evidence, not final proof.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    rows = build_rows()
    deltas = build_deltas(rows)

    summary_path = out_dir / "cross_model_text_effect_summary.csv"
    deltas_path = out_dir / "cross_model_text_effect_deltas.csv"
    report_path = out_dir / "CROSS_MODEL_TEXT_EFFECT_COMPARISON.md"

    write_csv(summary_path, rows)
    write_csv(deltas_path, deltas)
    write_report(report_path, rows, deltas)

    print(f"wrote {len(rows)} model rows to {summary_path}")
    print(f"wrote {len(deltas)} delta rows to {deltas_path}")
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
