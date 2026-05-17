#!/usr/bin/env python3
"""Merge document-level text features into the daily PPO price panel.

The merge is causal at day level: a document's available_at date is mapped to
the first trading date in the base panel on or after that date. Missing text
features are filled with zero. This script is for the toy handoff package; the
production project can use stricter intraday decision-time logic.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


TEXT_FEATURES = [
    "text_relevance_to_portfolio",
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


def _out_col(feature: str) -> str:
    return feature if feature.startswith("text_") else f"text_{feature}"


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _date_part(value: str) -> str:
    return (value or "")[:10]


def _panel_fieldnames(fieldnames: list[str] | None, text_fields: list[str] | None = None) -> list[str]:
    text_set = set(text_fields or [])
    return [
        field
        for field in (fieldnames or [])
        if field and not field.startswith("Unnamed") and field not in text_set
    ]


def _read_calendar_and_tickers(base_panel: Path) -> tuple[list[str], set[str]]:
    dates: set[str] = set()
    tickers: set[str] = set()
    with base_panel.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("date"):
                dates.add(row["date"][:10])
            if row.get("tic"):
                tickers.add(row["tic"].upper())
    return sorted(dates), tickers


def _next_trading_date(date_value: str, calendar: list[str]) -> str | None:
    if not date_value:
        return None
    idx = bisect.bisect_left(calendar, date_value[:10])
    if idx >= len(calendar):
        return None
    return calendar[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_feature_aggregates(features_csv: Path, calendar: list[str], tickers: set[str]):
    stock_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    market_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    doc_counts: dict[tuple[str, str], int] = defaultdict(int)
    market_counts: dict[str, int] = defaultdict(int)
    skipped = 0

    with features_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_date = row.get("decision_date") or _date_part(row.get("available_at", ""))
            trade_date = _next_trading_date(raw_date, calendar)
            if not trade_date:
                skipped += 1
                continue
            matched = [item.strip().upper() for item in (row.get("matched_tickers") or "").split("|") if item.strip()]
            is_market = not matched or "MARKET" in matched
            if is_market:
                for feature in TEXT_FEATURES:
                    market_values[trade_date][feature].append(_safe_float(row.get(feature, "0")))
                market_counts[trade_date] += 1
                continue
            used = False
            for ticker in matched:
                if ticker not in tickers:
                    continue
                for feature in TEXT_FEATURES:
                    stock_values[(trade_date, ticker)][feature].append(_safe_float(row.get(feature, "0")))
                doc_counts[(trade_date, ticker)] += 1
                used = True
            if not used:
                skipped += 1
    return stock_values, market_values, doc_counts, market_counts, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-panel", default="data/processed_final_fixed_external_lagclean_full.csv")
    parser.add_argument("--text-features", default="artifacts/text_features_mistral.csv")
    parser.add_argument("--output", default="artifacts/merged_panel_with_text.csv")
    parser.add_argument("--manifest", default="artifacts/merge_manifest.json")
    args = parser.parse_args()

    base_panel = Path(args.base_panel)
    features_csv = Path(args.text_features)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    calendar, tickers = _read_calendar_and_tickers(base_panel)
    stock_values, market_values, doc_counts, market_counts, skipped = _load_feature_aggregates(features_csv, calendar, tickers)

    output_fields: list[str] | None = None
    rows_written = 0
    text_matched_rows = 0
    with base_panel.open("r", encoding="utf-8-sig", newline="") as src, output.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        text_fields = [_out_col(name) for name in TEXT_FEATURES] + ["text_doc_count", "text_market_doc_count"]
        output_fields = _panel_fieldnames(reader.fieldnames, text_fields) + text_fields
        writer = csv.DictWriter(dst, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            date = (row.get("date") or "")[:10]
            ticker = (row.get("tic") or "").upper()
            key = (date, ticker)
            has_text = False
            for feature in TEXT_FEATURES:
                stock = stock_values.get(key, {}).get(feature, [])
                market = market_values.get(date, {}).get(feature, [])
                combined = stock + market
                row[_out_col(feature)] = f"{_mean(combined):.6f}"
                if combined:
                    has_text = True
            row["text_doc_count"] = str(doc_counts.get(key, 0))
            row["text_market_doc_count"] = str(market_counts.get(date, 0))
            if has_text:
                text_matched_rows += 1
            writer.writerow(row)
            rows_written += 1

    manifest = {
        "base_panel": str(base_panel),
        "text_features": str(features_csv),
        "output": str(output),
        "rows_written": rows_written,
        "calendar_dates": len(calendar),
        "tickers": len(tickers),
        "stock_date_ticker_feature_keys": len(stock_values),
        "market_feature_dates": len(market_values),
        "text_matched_rows": text_matched_rows,
        "skipped_feature_rows": skipped,
        "text_feature_columns": [_out_col(name) for name in TEXT_FEATURES],
        "missing_text_policy": "filled_with_zero",
        "date_policy": "available_at date mapped to first trading date >= available_at date",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
