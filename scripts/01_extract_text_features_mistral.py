#!/usr/bin/env python3
"""Extract the first text-to-numerical feature set with Mistral.

The script is deliberately dependency-light: it uses only the Python standard
library. Set MISTRAL_API_KEY in your environment or pass --api-key.

Use --dry-run to test the pipeline without spending API tokens. Dry-run writes a
simple deterministic rule-like baseline, not real LLM labels.
"""

from __future__ import annotations

import argparse
import csv
import glob
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


FEATURES: list[dict[str, Any]] = [
    {
        "name": "text_relevance_to_portfolio",
        "min": 0.0,
        "max": 1.0,
        "description": "How decision-relevant the document is for the ticker/portfolio.",
    },
    {
        "name": "sentiment_direction",
        "min": -1.0,
        "max": 1.0,
        "description": "Overall financial tone: negative -1, neutral 0, positive +1.",
    },
    {
        "name": "price_impact_direction",
        "min": -1.0,
        "max": 1.0,
        "description": "Expected directional pressure on the affected stock/portfolio.",
    },
    {
        "name": "risk_intensity",
        "min": 0.0,
        "max": 1.0,
        "description": "How strongly the document increases or discusses downside risk.",
    },
    {
        "name": "uncertainty_intensity",
        "min": 0.0,
        "max": 1.0,
        "description": "How uncertain/conditional the outlook is.",
    },
    {
        "name": "opportunity_intensity",
        "min": 0.0,
        "max": 1.0,
        "description": "How much upside, growth, demand, margin or capital-return opportunity is present.",
    },
    {
        "name": "forward_looking_intensity",
        "min": 0.0,
        "max": 1.0,
        "description": "How much guidance/outlook/future expectation content is present.",
    },
    {
        "name": "earnings_guidance_impact",
        "min": -1.0,
        "max": 1.0,
        "description": "Impact of earnings, revenue, margin, EPS or guidance content.",
    },
    {
        "name": "macro_financial_conditions_impact",
        "min": -1.0,
        "max": 1.0,
        "description": "Impact of rates, credit, inflation, volatility, labor or energy conditions.",
    },
    {
        "name": "company_event_risk_impact",
        "min": -1.0,
        "max": 1.0,
        "description": "Impact of legal, regulatory, supply-chain, demand, margin or company-specific risk.",
    },
]

OUTPUT_FIELDS = [
    "doc_id",
    "title",
    "available_at",
    "published_at",
    "decision_date",
    "matched_tickers",
    "source",
    "source_type",
    "split",
    "extractor_model",
    "extractor_status",
    *[item["name"] for item in FEATURES],
]

POSITIVE_TERMS = {
    "growth",
    "record",
    "increase",
    "increased",
    "strong",
    "beat",
    "beats",
    "raised",
    "higher",
    "improved",
    "buyback",
    "dividend",
    "margin expansion",
}

NEGATIVE_TERMS = {
    "decline",
    "decrease",
    "decreased",
    "weak",
    "miss",
    "missed",
    "lower",
    "loss",
    "litigation",
    "investigation",
    "risk",
    "uncertain",
    "pressure",
    "impairment",
}

RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RateLimitExceeded(RuntimeError):
    """Raised when the provider keeps returning 429 after retries."""

    pass


def _iter_jsonl(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.extend(Path(match) for match in glob.glob(item))
    return sorted(set(files))


def _safe_date(value: str) -> str:
    if not value:
        return ""
    return value[:10]


def _body_excerpt(row: dict[str, Any], max_chars: int) -> str:
    body = str(row.get("body") or "")
    body = re.sub(r"\s+", " ", body).strip()
    return body[:max_chars]


def _matched_tickers(row: dict[str, Any]) -> str:
    values = row.get("matched_tickers") or row.get("tickers_detected") or []
    if isinstance(values, str):
        return values
    return "|".join(str(item).upper() for item in values if item)


def _clamp(name: str, value: Any) -> float:
    spec = next(item for item in FEATURES if item["name"] == name)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(float(spec["min"]), min(float(spec["max"]), parsed))


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _retry_after_seconds(exc: BaseException, fallback: float, rate_limit_cooldown: float) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                return max(float(header), fallback)
            except ValueError:
                pass
        if exc.code == 429:
            return max(rate_limit_cooldown, fallback)
    return fallback


def _resume_rows(output: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not output.exists() or output.stat().st_size <= 0:
        return [], set()
    rows: list[dict[str, str]] = []
    completed: set[str] = set()
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = str(row.get("doc_id", "") or "")
            status = str(row.get("extractor_status", "") or "")
            if not doc_id or status != "ok":
                continue
            rows.append({field: str(row.get(field, "") or "") for field in OUTPUT_FIELDS})
            completed.add(doc_id)
    return rows, completed


def _dry_run_features(row: dict[str, Any]) -> dict[str, float]:
    text = (_body_excerpt(row, 6000) + " " + str(row.get("title") or "")).lower()
    pos = sum(1 for term in POSITIVE_TERMS if term in text)
    neg = sum(1 for term in NEGATIVE_TERMS if term in text)
    sentiment = max(-1.0, min(1.0, (pos - neg) / max(pos + neg, 3)))
    risk = min(1.0, neg / 8.0)
    opportunity = min(1.0, pos / 8.0)
    uncertainty = min(1.0, sum(text.count(term) for term in ("may", "could", "expect", "uncertain", "risk")) / 12.0)
    forward = min(1.0, sum(text.count(term) for term in ("guidance", "outlook", "expect", "forecast", "will")) / 10.0)
    earnings_terms = ("earnings", "revenue", "eps", "margin", "quarter", "guidance")
    macro_terms = ("rates", "inflation", "credit", "yield", "vix", "oil", "labor")
    company_risk_terms = ("litigation", "regulatory", "supply", "demand", "margin pressure", "investigation")
    return {
        "text_relevance_to_portfolio": 0.8 if _matched_tickers(row) else 0.5,
        "sentiment_direction": sentiment,
        "price_impact_direction": sentiment * (1.0 - 0.35 * risk),
        "risk_intensity": risk,
        "uncertainty_intensity": uncertainty,
        "opportunity_intensity": opportunity,
        "forward_looking_intensity": forward,
        "earnings_guidance_impact": sentiment if any(term in text for term in earnings_terms) else 0.0,
        "macro_financial_conditions_impact": sentiment if any(term in text for term in macro_terms) else 0.0,
        "company_event_risk_impact": -risk if any(term in text for term in company_risk_terms) else 0.0,
    }


def _build_prompt(row: dict[str, Any], max_chars: int) -> str:
    feature_lines = "\n".join(
        f'- "{item["name"]}" range [{item["min"]}, {item["max"]}]: {item["description"]}'
        for item in FEATURES
    )
    payload = {
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "source": row.get("source"),
        "source_type": row.get("source_type"),
        "available_at": row.get("available_at"),
        "matched_tickers": row.get("matched_tickers") or row.get("tickers_detected"),
        "body_excerpt": _body_excerpt(row, max_chars),
    }
    return (
        "You are extracting numeric financial text features for a PPO portfolio model.\n"
        "Return ONLY one valid JSON object. Do not include markdown.\n\n"
        "Fill these fields:\n"
        f"{feature_lines}\n\n"
        "Rules:\n"
        "- Use 0 when the document does not contain enough evidence for that feature.\n"
        "- Do not give trading advice.\n"
        "- Use the text only; do not use future price movement.\n"
        "- Keep values inside the specified ranges.\n\n"
        f"Document:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _call_mistral(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    retry_sleep_seconds: float,
    rate_limit_cooldown_seconds: float,
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract strict JSON numeric financial features."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            base_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return _parse_json_object(content)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_STATUS:
                raise
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc
        if attempt < max_retries:
            fallback_sleep = retry_sleep_seconds * (2**attempt)
            sleep_for = _retry_after_seconds(last_error, fallback_sleep, rate_limit_cooldown_seconds)
            print(
                f"Retry {attempt + 1}/{max_retries} after {type(last_error).__name__}: {last_error}. "
                f"Sleeping {sleep_for:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
        raise RateLimitExceeded("Mistral returned HTTP 429 after all retries. Stop now and resume later.") from last_error
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=["data/train_2010_2021", "data/test_2021_2023"])
    parser.add_argument("--output", default="artifacts/text_features_mistral.csv")
    parser.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"))
    parser.add_argument("--base-url", default=os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--rate-limit-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true", help="Ignore existing output and start from the first document.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = _iter_jsonl(args.input)
    if not files:
        raise SystemExit(f"No JSONL files found for input: {args.input}")
    if not args.dry_run and not args.api_key:
        raise SystemExit("MISTRAL_API_KEY is missing. Set it or use --dry-run.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_rows: list[dict[str, str]] = []
    completed_doc_ids: set[str] = set()
    if args.resume and not args.restart:
        resume_rows, completed_doc_ids = _resume_rows(output)
        if completed_doc_ids:
            print(f"Resuming from {output}: keeping {len(completed_doc_ids)} completed rows.")

    written = len(resume_rows)
    new_rows = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for existing in resume_rows:
            writer.writerow(existing)
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc_id = str(row.get("doc_id", "") or "")
                    if doc_id and doc_id in completed_doc_ids:
                        continue
                    if args.limit and written >= args.limit:
                        print(f"Wrote {written} rows to {output} ({new_rows} new rows)")
                        return 0
                    status = "ok"
                    try:
                        if args.dry_run:
                            values = _dry_run_features(row)
                            model_name = "dry_run_rule_like_stub"
                        else:
                            prompt = _build_prompt(row, args.max_body_chars)
                            values = _call_mistral(
                                prompt,
                                args.api_key,
                                args.model,
                                args.base_url,
                                args.timeout_seconds,
                                args.max_retries,
                                args.retry_sleep_seconds,
                                args.rate_limit_cooldown_seconds,
                            )
                            model_name = args.model
                    except RateLimitExceeded as exc:
                        print(
                            f"{exc} Completed rows are preserved in {output}. "
                            "Wait for the provider limit to reset and rerun the same command.",
                            file=sys.stderr,
                        )
                        print(f"Wrote {written} rows to {output} ({new_rows} new rows)")
                        return 75
                    except (
                        urllib.error.URLError,
                        urllib.error.HTTPError,
                        http.client.HTTPException,
                        TimeoutError,
                        ConnectionError,
                        OSError,
                        KeyError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        values = {item["name"]: 0.0 for item in FEATURES}
                        model_name = args.model
                        status = f"failed:{type(exc).__name__}"

                    out = {
                        "doc_id": row.get("doc_id", ""),
                        "title": row.get("title", ""),
                        "available_at": row.get("available_at", ""),
                        "published_at": row.get("published_at", ""),
                        "decision_date": _safe_date(row.get("available_at", "")),
                        "matched_tickers": _matched_tickers(row),
                        "source": row.get("source", ""),
                        "source_type": row.get("source_type", ""),
                        "split": row.get("split", ""),
                        "extractor_model": model_name,
                        "extractor_status": status,
                    }
                    for item in FEATURES:
                        name = item["name"]
                        out[name] = f"{_clamp(name, values.get(name, 0.0)):.6f}"
                    writer.writerow(out)
                    handle.flush()
                    written += 1
                    new_rows += 1
                    if not args.dry_run and args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
    print(f"Wrote {written} rows to {output} ({new_rows} new rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
