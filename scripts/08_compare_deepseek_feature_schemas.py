#!/usr/bin/env python
"""Compare candidate text feature schemas using only DeepSeek.

This is an extraction-quality diagnostic before expensive PPO training. It
answers: which feature schema produces valid, non-constant, source-aware signals
on the same sampled documents?
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import glob
import hashlib
import http.client
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_CONFIG = TOY_ROOT / "feature_schema_deepseek_candidates.json"
DEFAULT_OUTPUT_DIR = TOY_ROOT / "artifacts" / "deepseek_feature_schema_comparison"
DEFAULT_REPORT = TOY_ROOT / "reports" / "deepseek_feature_schema_comparison.md"
DEFAULT_INPUTS = [TOY_ROOT / "data" / "train_2010_2021", TOY_ROOT / "data" / "test_2021_2023"]
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def normalize_chat_completions_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if not url:
        return "https://llmapi.paratera.com/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if url == "https://api.deepseek.com":
        return f"{url}/chat/completions"
    return f"{url}/chat/completions"


def default_base_url() -> str:
    if os.environ.get("DEEPSEEK_BASE_URL"):
        return normalize_chat_completions_url(os.environ["DEEPSEEK_BASE_URL"])
    if os.environ.get("DEBATE_BASE_URL"):
        return normalize_chat_completions_url(os.environ["DEBATE_BASE_URL"])
    if os.environ.get("DEBATE_API_URL"):
        return normalize_chat_completions_url(f"{os.environ['DEBATE_API_URL'].rstrip('/')}/v1")
    return "https://llmapi.paratera.com/v1/chat/completions"


def default_model() -> str:
    return first_env("DEEPSEEK_MODEL", "DEBATE_STUDENT_MODEL") or "DeepSeek-V4-Flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=None)
    parser.add_argument("--schema-config", type=Path, default=DEFAULT_SCHEMA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--api-key", default=first_env("DEEPSEEK_API_KEY", "DEBATE_API_KEY", "OPENAI_API_KEY"))
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument("--docs-per-family", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--max-body-chars", type=int, default=6500)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--schemas", nargs="*", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true", help="Ignore previous prediction rows and overwrite outputs.")
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send OpenAI/DeepSeek JSON response_format. Disabled by default for wider OpenAI-compatible gateway support.",
    )
    parser.add_argument(
        "--fail-on-zero-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return a non-zero exit code if a live run produced zero successful predictions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TOY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def iter_jsonl(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        if item.is_dir():
            files.extend(sorted(item.glob("*.jsonl")))
        else:
            files.extend(Path(match) for match in glob.glob(str(item)))
    return sorted(set(files))


def pipe_join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item).upper() for item in value if item)
    return str(value)


def body_excerpt(row: dict[str, Any], max_chars: int) -> str:
    body = str(row.get("body") or "")
    body = re.sub(r"\s+", " ", body).strip()
    return body[:max_chars]


def source_family(row: dict[str, Any], path: Path) -> str:
    source_type = str(row.get("source_type") or "").lower()
    event_type = str(row.get("event_type") or "").lower()
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


def load_documents(paths: list[Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in iter_jsonl(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                docs.append(
                    {
                        "doc_id": row.get("doc_id", ""),
                        "document_hash": row.get("document_hash", ""),
                        "title": row.get("title", ""),
                        "available_at": row.get("available_at", ""),
                        "published_at": row.get("published_at", ""),
                        "matched_tickers": pipe_join(row.get("matched_tickers") or row.get("tickers_detected")),
                        "source": row.get("source", ""),
                        "source_type": row.get("source_type", ""),
                        "event_type": row.get("event_type", ""),
                        "split": row.get("split", ""),
                        "source_family": source_family(row, path),
                        "body_word_count": row.get("body_word_count", ""),
                        "body_excerpt": body_excerpt(row, 9000),
                    }
                )
    if not docs:
        raise ValueError(f"No documents found for inputs: {paths}")
    return docs


def stratified_sample(docs: list[dict[str, Any]], docs_per_family: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        by_family.setdefault(doc["source_family"], []).append(doc)
    sample: list[dict[str, Any]] = []
    for family in sorted(by_family):
        bucket = by_family[family]
        ticker_docs = [doc for doc in bucket if doc.get("matched_tickers")]
        macro_or_other = [doc for doc in bucket if not doc.get("matched_tickers")]
        rng.shuffle(ticker_docs)
        rng.shuffle(macro_or_other)
        chosen = (ticker_docs + macro_or_other)[:docs_per_family]
        sample.extend(chosen)
    return sample


def load_schemas(path: Path, names: list[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schemas = payload["schemas"]
    if names:
        keep = set(names)
        schemas = [schema for schema in schemas if schema["name"] in keep]
        missing = sorted(keep - {schema["name"] for schema in schemas})
        if missing:
            raise ValueError(f"Unknown schema names: {missing}")
    return schemas


def feature_specs(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in schema["features"]}


def prompt_for(schema: dict[str, Any], doc: dict[str, Any], max_body_chars: int) -> str:
    feature_lines = "\n".join(
        f'- "{item["name"]}" range [{item["min"]}, {item["max"]}]: {item["meaning"]}'
        for item in schema["features"]
    )
    payload = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "source": doc["source"],
        "source_type": doc["source_type"],
        "event_type": doc["event_type"],
        "source_family": doc["source_family"],
        "available_at": doc["available_at"],
        "published_at": doc["published_at"],
        "matched_tickers": doc["matched_tickers"],
        "body_excerpt": doc["body_excerpt"][:max_body_chars],
    }
    return (
        "You extract numeric financial text features for a causal PPO feature generator.\n"
        "Return ONLY one valid JSON object with this exact shape:\n"
        "{\"features\": {feature_name: number, ...}, \"confidence\": number, \"evidence_summary\": string}\n\n"
        f"Schema name: {schema['name']}\n"
        f"Schema purpose: {schema.get('purpose', '')}\n"
        "Fill every feature. Use 0 when evidence is absent. Do not infer from future prices.\n"
        "Penalize boilerplate: generic risk-factor language should not create a high actionable signal unless the text is specific.\n"
        "For macro documents, score portfolio-level macro evidence, not single-company events.\n\n"
        f"Features:\n{feature_lines}\n\n"
        f"Document:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def call_deepseek(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    json_mode: bool,
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON financial feature extractor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}
    last_error: BaseException | None = None
    retry_after: str | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            base_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return parse_json_object(payload["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After")
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                body = ""
            body = re.sub(r"\s+", " ", body).strip()[:400]
            message = f"HTTP {exc.code} {exc.reason}"
            if body:
                message = f"{message}: {body}"
            last_error = RuntimeError(message)
            if exc.code not in RETRYABLE_HTTP_STATUS:
                raise last_error from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
        if attempt < max_retries:
            sleep_for = 3.0 * (2**attempt)
            if retry_after:
                try:
                    sleep_for = max(sleep_for, float(retry_after))
                except ValueError:
                    pass
            print(f"Retry {attempt + 1}/{max_retries}: {type(last_error).__name__}: {last_error}", file=sys.stderr)
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error


def clamp(value: Any, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    if not math.isfinite(parsed):
        parsed = 0.0
    return max(minimum, min(maximum, parsed))


def dry_run_values(schema: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    text = f"{doc.get('title', '')} {doc.get('body_excerpt', '')}".lower()
    positive = sum(text.count(term) for term in ["growth", "record", "increase", "strong", "margin expansion", "buyback", "dividend"])
    negative = sum(text.count(term) for term in ["risk", "decline", "pressure", "litigation", "uncertain", "debt", "inflation", "cost"])
    numeric = len(re.findall(r"[-+]?\d+(?:\.\d+)?\s?(?:%|billion|million|bps|x|usd|dollars)?", text))
    values: dict[str, float] = {}
    for spec in schema["features"]:
        name = spec["name"]
        lo = float(spec["min"])
        hi = float(spec["max"])
        raw = 0.0
        if "risk" in name or "stress" in name or "pressure" in name:
            raw = min(1.0, negative / 12.0)
        elif "momentum" in name or "strength" in name or "support" in name or "opportunity" in name:
            raw = min(1.0, positive / 12.0)
        elif "quality" in name or "confidence" in name or "specificity" in name:
            raw = min(1.0, numeric / 20.0)
        elif "boilerplate" in name:
            raw = min(1.0, text.count("risk factors") / 3.0)
        elif lo < 0:
            raw = max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 4)))
        values[name] = clamp(raw, lo, hi)
    return {"features": values, "confidence": 0.5, "evidence_summary": "dry-run keyword diagnostic"}


def prediction_key(schema_name: str, doc_id: str) -> str:
    return f"{schema_name}::{doc_id}"


def load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
                keys.add(prediction_key(row["schema_name"], row["doc_id"]))
    return rows, keys


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_extraction(args: argparse.Namespace, schemas: list[dict[str, Any]], docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_path = args.output_dir / "sample_predictions.jsonl"
    existing_rows, completed = load_existing(predictions_path) if args.resume and not args.restart else ([], set())
    rows = existing_rows[:]
    if not args.dry_run and not args.api_key:
        raise SystemExit(
            "API key is missing. Set DEEPSEEK_API_KEY, DEBATE_API_KEY, or OPENAI_API_KEY; "
            "pass --api-key; or run --dry-run."
        )

    for schema in schemas:
        specs = feature_specs(schema)
        for doc in docs:
            key = prediction_key(schema["name"], doc["doc_id"])
            if key in completed:
                continue
            status = "ok"
            error = ""
            try:
                if args.dry_run:
                    payload = dry_run_values(schema, doc)
                    model_name = "dry_run_keyword_stub"
                else:
                    payload = call_deepseek(
                        prompt_for(schema, doc, args.max_body_chars),
                        args.api_key,
                        args.model,
                        args.base_url,
                        args.timeout_seconds,
                        args.max_retries,
                        args.json_mode,
                    )
                    model_name = args.model
                raw_features = payload.get("features", payload)
                values = {
                    name: clamp(raw_features.get(name, 0.0), float(spec["min"]), float(spec["max"]))
                    for name, spec in specs.items()
                }
                confidence = clamp(payload.get("confidence", 0.0), 0.0, 1.0)
                evidence_summary = str(payload.get("evidence_summary", ""))[:500]
            except Exception as exc:  # keep resumable extraction robust
                status = f"failed:{type(exc).__name__}"
                error = str(exc)[:500]
                model_name = args.model
                values = {name: 0.0 for name in specs}
                confidence = 0.0
                evidence_summary = ""
            row = {
                "schema_name": schema["name"],
                "doc_id": doc["doc_id"],
                "document_hash": doc["document_hash"],
                "title": doc["title"],
                "source_family": doc["source_family"],
                "source_type": doc["source_type"],
                "event_type": doc["event_type"],
                "split": doc["split"],
                "available_at": doc["available_at"],
                "matched_tickers": doc["matched_tickers"],
                "extractor_model": model_name,
                "status": status,
                "confidence": confidence,
                "evidence_summary": evidence_summary,
                "features": values,
                "error": error,
            }
            rows.append(row)
            write_jsonl(predictions_path, rows)
            if status == "ok":
                completed.add(key)
            if not args.dry_run and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    return rows


def flatten_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        for feature, value in row.get("features", {}).items():
            flat.append(
                {
                    "schema_name": row["schema_name"],
                    "doc_id": row["doc_id"],
                    "source_family": row["source_family"],
                    "feature": feature,
                    "value": float(value),
                    "abs_value": abs(float(value)),
                    "nonzero": abs(float(value)) > 1e-9,
                    "status": row["status"],
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                }
            )
    return flat


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def schema_summaries(rows: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = flatten_predictions([row for row in rows if row.get("status") == "ok"])
    out: list[dict[str, Any]] = []
    expected_docs_by_schema = {schema["name"]: 0 for schema in schemas}
    for row in rows:
        expected_docs_by_schema[row["schema_name"]] += 1
    for schema in schemas:
        name = schema["name"]
        schema_rows = [row for row in rows if row["schema_name"] == name]
        ok_docs = [row for row in schema_rows if row["status"] == "ok"]
        flat_rows = [row for row in flat if row["schema_name"] == name]
        values = [row["value"] for row in flat_rows]
        abs_values = [abs(value) for value in values]
        feature_stds = []
        low_variance = 0
        sparse = 0
        for feature in [item["name"] for item in schema["features"]]:
            feature_values = [row["value"] for row in flat_rows if row["feature"] == feature]
            feature_std = stdev(feature_values)
            feature_stds.append(feature_std)
            if feature_std < 0.02:
                low_variance += 1
            nonzero_share = mean([1.0 if abs(v) > 1e-9 else 0.0 for v in feature_values])
            if nonzero_share < 0.05:
                sparse += 1
        source_nonzero = []
        for family in sorted({row["source_family"] for row in flat_rows}):
            family_rows = [row for row in flat_rows if row["source_family"] == family]
            source_nonzero.append(mean([1.0 if row["nonzero"] else 0.0 for row in family_rows]))
        source_balance = 1.0 - min(1.0, stdev(source_nonzero)) if source_nonzero else 0.0
        validity = len(ok_docs) / len(schema_rows) if schema_rows else 0.0
        nonzero = mean([1.0 if abs(value) > 1e-9 else 0.0 for value in values])
        avg_std = mean(feature_stds)
        confidence = mean([float(row.get("confidence", 0.0) or 0.0) for row in ok_docs])
        utility_score = (
            0.30 * validity
            + 0.25 * min(avg_std / 0.25, 1.0)
            + 0.20 * (1.0 - abs(nonzero - 0.45) / 0.45)
            + 0.15 * source_balance
            + 0.10 * confidence
        )
        out.append(
            {
                "schema_name": name,
                "feature_count": len(schema["features"]),
                "documents": len(schema_rows),
                "ok_documents": len(ok_docs),
                "validity_rate": validity,
                "nonzero_share": nonzero,
                "mean_abs_value": mean(abs_values),
                "avg_feature_std": avg_std,
                "low_variance_feature_count": low_variance,
                "sparse_feature_count": sparse,
                "source_balance_score": source_balance,
                "mean_confidence": confidence,
                "diagnostic_utility_score": max(0.0, min(1.0, utility_score)),
            }
        )
    return sorted(out, key=lambda row: row["diagnostic_utility_score"], reverse=True)


def feature_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = flatten_predictions([row for row in rows if row.get("status") == "ok"])
    keys = sorted({(row["schema_name"], row["feature"]) for row in flat})
    out = []
    for schema_name, feature in keys:
        feature_rows = [row for row in flat if row["schema_name"] == schema_name and row["feature"] == feature]
        values = [row["value"] for row in feature_rows]
        out.append(
            {
                "schema_name": schema_name,
                "feature": feature,
                "documents": len(feature_rows),
                "mean_value": mean(values),
                "std_value": stdev(values),
                "nonzero_share": mean([1.0 if abs(value) > 1e-9 else 0.0 for value in values]),
                "mean_abs_value": mean([abs(value) for value in values]),
            }
        )
    return out


def source_family_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = flatten_predictions([row for row in rows if row.get("status") == "ok"])
    keys = sorted({(row["schema_name"], row["source_family"]) for row in flat})
    out = []
    for schema_name, family in keys:
        family_rows = [row for row in flat if row["schema_name"] == schema_name and row["source_family"] == family]
        values = [row["value"] for row in family_rows]
        out.append(
            {
                "schema_name": schema_name,
                "source_family": family,
                "values": len(values),
                "nonzero_share": mean([1.0 if abs(value) > 1e-9 else 0.0 for value in values]),
                "mean_abs_value": mean([abs(value) for value in values]),
                "avg_feature_std": stdev(values),
            }
        )
    return out


def error_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    error_rows = [row for row in rows if row.get("status") != "ok"]
    counts = Counter(
        (
            row.get("schema_name", ""),
            row.get("status", ""),
            str(row.get("error", "")).replace("\n", " ")[:180],
        )
        for row in error_rows
    )
    return [
        {
            "schema_name": schema_name,
            "status": status,
            "count": count,
            "example_error": error,
        }
        for (schema_name, status, error), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 12) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:max_rows]:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    schema_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# DeepSeek Feature Schema Comparison",
        "",
        "Goal: compare new text feature schemas using one DeepSeek extractor and the same stratified document sample.",
        "",
        "This is an extraction diagnostic, not a PPO performance claim.",
        "",
        "## Schema ranking",
        "",
        md_table(
            schema_rows,
            [
                "schema_name",
                "feature_count",
                "ok_documents",
                "validity_rate",
                "nonzero_share",
                "avg_feature_std",
                "low_variance_feature_count",
                "diagnostic_utility_score",
            ],
        ),
        "",
        "## Source-family behavior",
        "",
        md_table(source_rows, ["schema_name", "source_family", "nonzero_share", "mean_abs_value", "avg_feature_std"], 20),
        "",
        "## Extraction errors",
        "",
        md_table(error_rows, ["schema_name", "status", "count", "example_error"], 12),
        "",
        "## How to read this",
        "",
        "- High validity means DeepSeek returned schema-valid JSON.",
        "- Very low nonzero share means the schema is too sparse for a daily PPO panel.",
        "- Very low feature std means the feature is nearly constant and likely useless.",
        "- Source-family imbalance means the schema may only work for one document type.",
        "- If every live request failed with 401, the API key, base URL, and model name do not belong to the same provider.",
        "- The best next PPO candidate should be compact, non-constant, and not dominated by boilerplate.",
        "",
        "```json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.base_url = normalize_chat_completions_url(args.base_url)
    if args.restart:
        args.resume = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.input if args.input else DEFAULT_INPUTS
    docs = stratified_sample(load_documents(inputs), args.docs_per_family, args.sample_seed)
    schemas = load_schemas(args.schema_config, args.schemas)

    sample_path = args.output_dir / "sample_docs.jsonl"
    sample_rows = [
        {key: value for key, value in doc.items() if key != "body_excerpt"}
        | {"body_excerpt_hash": hashlib.sha256(doc["body_excerpt"].encode("utf-8")).hexdigest()}
        for doc in docs
    ]
    write_jsonl(sample_path, sample_rows)

    prediction_rows = run_extraction(args, schemas, docs)
    ok_rows = [row for row in prediction_rows if row.get("status") == "ok"]
    error_rows = [row for row in prediction_rows if row.get("status") != "ok"]
    errors_path = args.output_dir / "extraction_errors.jsonl"
    write_jsonl(errors_path, error_rows)

    schema_rows = schema_summaries(prediction_rows, schemas)
    feature_rows = feature_summaries(prediction_rows)
    source_rows = source_family_summaries(prediction_rows)
    error_summary_rows = error_summaries(prediction_rows)
    schema_path = args.output_dir / "schema_summary.csv"
    feature_path = args.output_dir / "feature_summary.csv"
    source_path = args.output_dir / "source_family_summary.csv"
    error_summary_path = args.output_dir / "error_summary.csv"
    write_csv(schema_path, schema_rows)
    write_csv(feature_path, feature_rows)
    write_csv(source_path, source_rows)
    write_csv(error_summary_path, error_summary_rows)

    status = "completed"
    if not args.dry_run and not ok_rows:
        status = "failed_no_successful_predictions"

    manifest = {
        "status": status,
        "dry_run": bool(args.dry_run),
        "model": "dry_run_keyword_stub" if args.dry_run else args.model,
        "base_url": args.base_url if not args.dry_run else "none",
        "json_mode": bool(args.json_mode),
        "schema_config": portable_path(args.schema_config),
        "output_dir": portable_path(args.output_dir),
        "report": portable_path(args.report),
        "sample_docs": portable_path(sample_path),
        "sample_doc_count": len(docs),
        "schemas": [schema["name"] for schema in schemas],
        "predictions": portable_path(args.output_dir / "sample_predictions.jsonl"),
        "ok_predictions": len(ok_rows),
        "error_predictions": len(error_rows),
        "schema_summary": portable_path(schema_path),
        "feature_summary": portable_path(feature_path),
        "source_family_summary": portable_path(source_path),
        "error_summary": portable_path(error_summary_path),
        "errors": portable_path(errors_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.report, args, schema_rows, source_rows, error_summary_rows, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if status == "failed_no_successful_predictions" and args.fail_on_zero_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
