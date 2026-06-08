#!/usr/bin/env python
"""Score document/source quality for the DRL lane.

This script joins cached document metadata with extracted text features and a
small event-style next-day return diagnostic. It does not train PPO; it creates
the source quality table needed before deciding which document families to
keep, downweight, or inspect manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = TOY_ROOT / "artifacts" / "document_source_quality"
DEFAULT_REPORT = TOY_ROOT / "reports" / "drl_document_source_quality.md"
DEFAULT_FEATURES = TOY_ROOT / "artifacts" / "text_features_mistral.csv"
DEFAULT_PANEL = TOY_ROOT / "artifacts" / "processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv"
DEFAULT_DOC_DIRS = [TOY_ROOT / "data" / "train_2010_2021", TOY_ROOT / "data" / "test_2021_2023"]

SIGNAL_COLUMNS = [
    "sentiment_direction",
    "price_impact_direction",
    "risk_intensity",
    "uncertainty_intensity",
    "opportunity_intensity",
    "forward_looking_intensity",
    "earnings_guidance_impact",
    "macro_financial_conditions_impact",
    "company_event_risk_impact",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--doc-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path}:{line_no}") from exc


def load_documents(doc_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc_dir in doc_dirs:
        for path in sorted(doc_dir.glob("*.jsonl")):
            for obj in iter_jsonl(path):
                rows.append(
                    {
                        "doc_id": obj.get("doc_id"),
                        "title": obj.get("title", ""),
                        "source": obj.get("source", ""),
                        "source_type": obj.get("source_type", ""),
                        "source_family": source_family(obj, path),
                        "split": obj.get("split", doc_dir.name),
                        "available_at": obj.get("available_at", ""),
                        "published_at": obj.get("published_at", ""),
                        "document_hash": obj.get("document_hash", ""),
                        "matched_tickers": pipe_join(obj.get("matched_tickers")),
                        "event_type": obj.get("event_type", ""),
                        "event_tags": pipe_join(obj.get("event_tags")),
                        "body_word_count": obj.get("body_word_count", np.nan),
                        "fetch_status": obj.get("fetch_status", ""),
                        "source_reliability_tier": obj.get("source_reliability_tier", ""),
                    }
                )
    if not rows:
        raise ValueError(f"No documents found under: {doc_dirs}")
    return pd.DataFrame(rows).drop_duplicates(subset=["doc_id"])


def pipe_join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    return str(value)


def source_family(obj: dict[str, Any], path: Path) -> str:
    source_type = str(obj.get("source_type") or "").lower()
    event_type = str(obj.get("event_type") or "").lower()
    file_name = path.name.lower()
    if "macro" in source_type or "macro" in event_type or "macro" in file_name:
        return "official_macro"
    if "sec" in source_type or "filing" in event_type or "sec_filing" in file_name:
        if "exhibit" in event_type or "exhibit" in source_type:
            return "sec_exhibit"
        return "sec_filing_section"
    if "earnings" in source_type or "earnings" in event_type:
        return "company_earnings_release"
    if "company" in source_type or "company_ir" in file_name:
        return "company_ir"
    return source_type or "unknown"


def reliability_score(row: pd.Series) -> float:
    family = str(row.get("source_family", "")).lower()
    tier = str(row.get("source_reliability_tier", "")).lower()
    source = str(row.get("source", "")).lower()
    if "sec" in family:
        return 1.0
    if "official_macro" in family or "fred" in source:
        return 0.95
    if "earnings" in family:
        return 0.9
    if "company_ir" in family or "official" in tier:
        return 0.85
    return 0.6


def timestamp_integrity(row: pd.Series) -> float:
    available = pd.to_datetime(row.get("available_at"), errors="coerce", utc=True)
    published = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
    has_hash = bool(str(row.get("document_hash", "")).strip())
    score = 0.0
    if pd.notna(available):
        score += 0.45
    if pd.notna(published):
        score += 0.25
        if pd.notna(available) and available >= published:
            score += 0.15
    if has_hash:
        score += 0.15
    return min(score, 1.0)


def add_next_day_returns(docs: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        docs["next_day_signed_return"] = np.nan
        docs["next_day_abs_return"] = np.nan
        return docs
    panel = pd.read_csv(panel_path, usecols=["date", "tic", "daily_return"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel.sort_values(["tic", "date"]).reset_index(drop=True)
    panel["next_day_signed_return"] = panel.groupby("tic")["daily_return"].shift(-1)
    lookup = panel.set_index(["tic", "date"])["next_day_signed_return"].to_dict()

    signed: list[float] = []
    for _, row in docs.iterrows():
        date_value = pd.to_datetime(row.get("decision_date") or row.get("available_at"), errors="coerce")
        tickers = [item.strip().upper() for item in str(row.get("matched_tickers", "")).split("|") if item.strip()]
        if pd.isna(date_value) or not tickers or "MARKET" in tickers:
            signed.append(np.nan)
            continue
        values = [lookup.get((ticker, date_value.date())) for ticker in tickers]
        values = [float(v) for v in values if v is not None and pd.notna(v)]
        signed.append(float(np.mean(values)) if values else np.nan)
    docs["next_day_signed_return"] = signed
    docs["next_day_abs_return"] = docs["next_day_signed_return"].abs()
    return docs


def score_documents(docs: pd.DataFrame, features: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    features = features.copy()
    for col in ["text_relevance_to_portfolio", *SIGNAL_COLUMNS]:
        features[col] = pd.to_numeric(features.get(col, 0.0), errors="coerce").fillna(0.0)
    features["signal_density"] = features[SIGNAL_COLUMNS].abs().mean(axis=1).clip(0.0, 1.0)
    features["extraction_ok"] = (features.get("extractor_status", "") == "ok").astype(float)

    cols = [
        "doc_id",
        "decision_date",
        "extractor_status",
        "text_relevance_to_portfolio",
        "signal_density",
        "extraction_ok",
        *SIGNAL_COLUMNS,
    ]
    merged = docs.merge(features[[col for col in cols if col in features.columns]], on="doc_id", how="left")
    merged["text_relevance_to_portfolio"] = merged["text_relevance_to_portfolio"].fillna(0.0)
    merged["signal_density"] = merged["signal_density"].fillna(0.0)
    merged["extraction_ok"] = merged["extraction_ok"].fillna(0.0)
    merged["timestamp_integrity"] = merged.apply(timestamp_integrity, axis=1)
    merged["source_reliability"] = merged.apply(reliability_score, axis=1)
    merged = add_next_day_returns(merged, panel_path)
    merged["source_quality_score"] = (
        0.30 * merged["text_relevance_to_portfolio"]
        + 0.25 * merged["signal_density"]
        + 0.20 * merged["timestamp_integrity"]
        + 0.15 * merged["source_reliability"]
        + 0.10 * merged["extraction_ok"]
    ).clip(0.0, 1.0)
    merged["recommendation"] = np.select(
        [
            merged["source_quality_score"] >= 0.70,
            merged["source_quality_score"] >= 0.45,
        ],
        ["keep_or_upweight", "inspect_or_downweight"],
        default="downweight_or_drop",
    )
    return merged.sort_values(["source_quality_score", "signal_density"], ascending=False).reset_index(drop=True)


def aggregate_sources(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored.groupby("source_family", dropna=False)
        .agg(
            documents=("doc_id", "count"),
            quality_mean=("source_quality_score", "mean"),
            relevance_mean=("text_relevance_to_portfolio", "mean"),
            signal_density_mean=("signal_density", "mean"),
            timestamp_integrity_mean=("timestamp_integrity", "mean"),
            source_reliability_mean=("source_reliability", "mean"),
            extraction_ok_rate=("extraction_ok", "mean"),
            next_day_abs_return_mean=("next_day_abs_return", "mean"),
        )
        .reset_index()
    )
    grouped["noise_flag"] = np.select(
        [
            grouped["signal_density_mean"] < 0.20,
            grouped["next_day_abs_return_mean"].isna(),
        ],
        ["low_text_signal_density", "no_ticker_event_diagnostic"],
        default="none",
    )
    grouped["recommendation"] = np.select(
        [
            (grouped["quality_mean"] >= 0.70) & (grouped["noise_flag"] == "none"),
            (grouped["quality_mean"] >= 0.70),
            grouped["quality_mean"] >= 0.45,
        ],
        ["keep_or_upweight", "keep_but_downweight_for_directional_signal", "inspect_or_downweight"],
        default="downweight_or_drop",
    )
    return grouped.sort_values("quality_mean", ascending=False).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    frame = df.head(max_rows).copy()
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, source_summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = source_summary[
        [
            "source_family",
            "documents",
            "quality_mean",
            "signal_density_mean",
            "timestamp_integrity_mean",
            "next_day_abs_return_mean",
            "noise_flag",
            "recommendation",
        ]
    ]
    lines = [
        "# DRL Issue #7: Document and source quality",
        "",
        "Goal: identify which document families produce usable text signals before PPO consumes them.",
        "",
        "## Source-family quality",
        "",
        markdown_table(compact),
        "",
        "## Scoring formula",
        "",
        "`quality = 0.30 relevance + 0.25 signal_density + 0.20 timestamp_integrity + 0.15 source_reliability + 0.10 extraction_ok`",
        "",
        "The next-day return column is an event-style diagnostic, not an alpha claim.",
        "PPO impact should be added after the ablation matrix from Issue #6 has trained runs.",
        "",
        "```json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TOY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    doc_dirs = args.doc_dir if args.doc_dir else DEFAULT_DOC_DIRS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    docs = load_documents(doc_dirs)
    features = pd.read_csv(args.features)
    scored = score_documents(docs, features, args.panel)
    source_summary = aggregate_sources(scored)

    document_path = args.output_dir / "document_quality.csv"
    source_path = args.output_dir / "source_family_quality.csv"
    scored.to_csv(document_path, index=False)
    source_summary.to_csv(source_path, index=False)
    manifest = {
        "status": "completed",
        "doc_dirs": [portable_path(path) for path in doc_dirs],
        "features": portable_path(args.features),
        "panel": portable_path(args.panel),
        "document_quality": portable_path(document_path),
        "source_family_quality": portable_path(source_path),
        "report": portable_path(args.report),
        "documents_scored": int(len(scored)),
        "source_families": int(source_summary["source_family"].nunique()),
        "ppo_impact_note": "PPO impact requires trained ablations from Issue #6; this script adds event-style next-day return diagnostics now.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.report, source_summary, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
