#!/usr/bin/env python3
"""Extract DeepSeek v2 PPO text features over the full document corpus.

This script is intentionally dependency-light and OpenAI-compatible. It keeps
the same causal handoff fields as the v1 extractor: available_at, decision_date,
matched_tickers, split, source metadata, and extractor status.
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
from pathlib import Path
from typing import Any


TOY_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = TOY_ROOT / ".env"


def _load_local_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_load_local_env()

DEFAULT_SCHEMA = TOY_ROOT / "feature_schema_deepseek_v2_ppo_compact.json"
DEFAULT_OUTPUT = TOY_ROOT / "artifacts" / "text_features_deepseek_v2.csv"
DEFAULT_MANIFEST = TOY_ROOT / "artifacts" / "text_features_deepseek_v2_manifest.json"
DEFAULT_INPUTS = [str(TOY_ROOT / "data" / "train_2010_2021"), str(TOY_ROOT / "data" / "test_2021_2023")]
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = (
    os.environ.get("LLM_BASE_URL")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or "https://llmapi.paratera.com/v1/chat/completions"
)
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
META_FIELDS = [
    "doc_id",
    "document_hash",
    "title",
    "available_at",
    "published_at",
    "decision_date",
    "matched_tickers",
    "source",
    "source_type",
    "source_family",
    "event_type",
    "split",
    "extractor_model",
    "extractor_status",
    "extractor_confidence",
    "evidence_summary",
]


class RateLimitExceeded(RuntimeError):
    """Raised when the provider keeps returning 429 after retries."""


def _api_key_from_env() -> str:
    for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "DEBATE_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _chat_completions_url(value: str) -> str:
    value = value.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def _iter_jsonl(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.extend(Path(match) for match in glob.glob(item))
    return sorted(set(files))


def _safe_date(value: Any) -> str:
    text = str(value or "")
    return text[:10] if text else ""


def _clean_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def _matched_tickers(row: dict[str, Any]) -> str:
    values = row.get("matched_tickers") or row.get("tickers_detected") or []
    if isinstance(values, str):
        return values
    return "|".join(str(item).upper() for item in values if item)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _feature_range(spec: dict[str, Any]) -> tuple[float, float]:
    if "range" in spec:
        lo, hi = spec["range"]
        return float(lo), float(hi)
    return float(spec.get("min", 0.0)), float(spec.get("max", 1.0))


def _load_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Schema has no features: {path}")
    for spec in features:
        if not spec.get("name"):
            raise ValueError(f"Schema feature is missing name: {spec}")
        _feature_range(spec)
    return payload


def _clamp(spec: dict[str, Any], value: Any) -> float:
    lo, hi = _feature_range(spec)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(lo, min(hi, parsed))


def _output_fields(schema: dict[str, Any]) -> list[str]:
    return [*META_FIELDS, *[item["name"] for item in schema["features"]]]


def _resume_rows(output: Path, expected_fields: list[str]) -> tuple[list[dict[str, str]], set[str]]:
    if not output.exists() or output.stat().st_size <= 0:
        return [], set()
    rows: list[dict[str, str]] = []
    completed: set[str] = set()
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != expected_fields:
            raise ValueError(
                f"Existing output header does not match schema. Use --restart or choose a new output path: {output}"
            )
        for row in reader:
            doc_id = str(row.get("doc_id", "") or "")
            status = str(row.get("extractor_status", "") or "")
            if not doc_id or status != "ok":
                continue
            rows.append({field: str(row.get(field, "") or "") for field in expected_fields})
            completed.add(doc_id)
    return rows, completed


def _build_prompt(schema: dict[str, Any], row: dict[str, Any], max_body_chars: int) -> str:
    feature_lines = []
    for spec in schema["features"]:
        lo, hi = _feature_range(spec)
        meaning = spec.get("meaning") or spec.get("description") or ""
        feature_lines.append(f'- "{spec["name"]}" range [{lo}, {hi}]: {meaning}')
    document = {
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "source": row.get("source"),
        "source_type": row.get("source_type"),
        "source_family": row.get("source_family"),
        "event_type": row.get("event_type"),
        "available_at": row.get("available_at"),
        "matched_tickers": row.get("matched_tickers") or row.get("tickers_detected"),
        "body_excerpt": _clean_text(row.get("body"), max_body_chars),
    }
    return (
        "You extract numeric financial text features for a PPO portfolio model.\n"
        "Return ONLY one valid JSON object. Do not include markdown.\n\n"
        "Return this schema:\n"
        "{\n"
        '  "confidence": number from 0 to 1,\n'
        '  "evidence_summary": short text explaining the evidence,\n'
        '  "features": { feature_name: numeric_value }\n'
        "}\n\n"
        "Feature definitions:\n"
        + "\n".join(feature_lines)
        + "\n\nRules:\n"
        "- Use 0 when the document lacks enough evidence for a feature.\n"
        "- Use only the supplied text; do not infer from future prices.\n"
        "- Do not give trading advice.\n"
        "- Keep every value inside its range.\n"
        "- Penalize generic administrative announcements with low confidence and mostly zero values.\n\n"
        f"Document:\n{json.dumps(document, ensure_ascii=False)}"
    )


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


def _call_chat_completion(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    max_retries: int,
    retry_sleep_seconds: float,
    rate_limit_cooldown_seconds: float,
    json_mode: bool,
) -> dict[str, Any]:
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract strict JSON numeric financial features."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            _chat_completions_url(base_url),
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
            return _parse_json_object(payload["choices"][0]["message"]["content"])
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
        raise RateLimitExceeded("Provider returned HTTP 429 after all retries. Stop now and resume later.") from last_error
    raise last_error


def _redact_api_error(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-...", str(text or ""))[:500]


def _api_preflight(
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    json_mode: bool,
) -> None:
    try:
        _call_chat_completion(
            'Return exactly this JSON object: {"ok": true}',
            api_key,
            model,
            base_url,
            timeout,
            max_retries=0,
            retry_sleep_seconds=0.0,
            rate_limit_cooldown_seconds=0.0,
            json_mode=json_mode,
        )
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        raise SystemExit(
            f"API preflight failed before writing output: HTTP {exc.code} {exc.reason}. "
            f"{_redact_api_error(body)}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"API preflight failed before writing output: {type(exc).__name__}: {_redact_api_error(str(exc))}") from exc


def _dry_run_payload(schema: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    text = (_clean_text(row.get("body"), 6000) + " " + str(row.get("title") or "")).lower()
    positive = sum(text.count(term) for term in ("growth", "record", "increase", "strong", "raised", "beat"))
    negative = sum(text.count(term) for term in ("risk", "decline", "weak", "pressure", "loss", "uncertain"))
    alpha = max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 3)))
    values: dict[str, float] = {}
    for spec in schema["features"]:
        name = spec["name"]
        if name == "text_alpha_direction":
            value = alpha
        elif name in {"text_downside_risk", "text_uncertainty", "text_macro_stress", "text_balance_sheet_stress"}:
            value = min(1.0, negative / 10.0)
        elif name == "text_earnings_pressure":
            value = alpha if any(term in text for term in ("earnings", "revenue", "eps", "margin", "guidance")) else 0.0
        elif name == "text_signal_confidence":
            value = 0.8 if _matched_tickers(row) else 0.35
        elif name == "text_evidence_specificity":
            value = 0.7 if re.search(r"\d", text) else 0.25
        elif name == "text_numeric_evidence_density":
            value = min(1.0, len(re.findall(r"\d", text)) / 120.0)
        elif name == "text_boilerplate_intensity":
            value = 0.6 if "forward-looking" in text or "safe harbor" in text else 0.15
        else:
            value = 0.0
        values[name] = _clamp(spec, value)
    return {
        "confidence": values.get("text_signal_confidence", 0.5),
        "evidence_summary": "dry-run deterministic text heuristic",
        "features": values,
    }


def _normalise_payload(schema: dict[str, Any], payload: dict[str, Any]) -> tuple[float, str, dict[str, float]]:
    raw_features = payload.get("features", payload)
    if not isinstance(raw_features, dict):
        raw_features = {}
    values: dict[str, float] = {}
    for spec in schema["features"]:
        name = spec["name"]
        values[name] = _clamp(spec, raw_features.get(name, 0.0))
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", values.get("text_signal_confidence", 0.0)))))
    except (TypeError, ValueError):
        confidence = values.get("text_signal_confidence", 0.0)
    summary = _clean_text(payload.get("evidence_summary", ""), 800)
    return confidence, summary, values


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TOY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=DEFAULT_INPUTS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api-key", default=_api_key_from_env())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--rate-limit-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-body-chars", type=int, default=6000)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=25,
        help="Stop after this many consecutive extraction failures; prevents wasting a long run during provider outages.",
    )
    parser.add_argument("--no-api-preflight", action="store_true", help="Skip the one-request API validation before writing output.")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true", help="Ignore existing output and start from the first document.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    schema = _load_schema(args.schema)
    fields = _output_fields(schema)
    files = _iter_jsonl(args.input)
    if not files:
        raise SystemExit(f"No JSONL files found for input: {args.input}")
    if not args.dry_run and not args.api_key:
        raise SystemExit("DeepSeek-compatible API key is missing. Set LLM_API_KEY/DEEPSEEK_API_KEY or use --dry-run.")
    if not args.dry_run and not args.no_api_preflight:
        _api_preflight(args.api_key, args.model, args.base_url, args.timeout_seconds, args.json_mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    resume_rows: list[dict[str, str]] = []
    completed_doc_ids: set[str] = set()
    if args.resume and not args.restart:
        resume_rows, completed_doc_ids = _resume_rows(args.output, fields)
        if completed_doc_ids:
            print(f"Resuming from {args.output}: keeping {len(completed_doc_ids)} completed rows.")

    written = len(resume_rows)
    new_rows = 0
    failed_rows = 0
    consecutive_failures = 0
    stop_reason = ""
    started_at = time.time()
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for existing in resume_rows:
            writer.writerow(existing)
        for file_path in files:
            if stop_reason:
                break
            with file_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc_id = str(row.get("doc_id", "") or "")
                    if doc_id and doc_id in completed_doc_ids:
                        continue
                    if args.limit and written >= args.limit:
                        break
                    try:
                        if args.dry_run:
                            payload = _dry_run_payload(schema, row)
                            model_name = "dry_run_rule_like_stub"
                        else:
                            payload = _call_chat_completion(
                                _build_prompt(schema, row, args.max_body_chars),
                                args.api_key,
                                args.model,
                                args.base_url,
                                args.timeout_seconds,
                                args.max_retries,
                                args.retry_sleep_seconds,
                                args.rate_limit_cooldown_seconds,
                                args.json_mode,
                            )
                            model_name = args.model
                        status = "ok"
                    except RateLimitExceeded as exc:
                        print(
                            f"{exc} Completed rows are preserved in {args.output}. "
                            "Wait for the provider limit to reset and rerun the same command.",
                            file=sys.stderr,
                        )
                        stop_reason = "rate_limit"
                        break
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
                        payload = {"confidence": 0.0, "evidence_summary": "", "features": {}}
                        model_name = args.model
                        status = f"failed:{type(exc).__name__}"
                        if isinstance(exc, urllib.error.HTTPError):
                            status = f"{status}:{exc.code}"
                        failed_rows += 1
                    if status == "ok":
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                    confidence, evidence_summary, values = _normalise_payload(schema, payload)
                    out = {
                        "doc_id": row.get("doc_id", ""),
                        "document_hash": row.get("document_hash", ""),
                        "title": row.get("title", ""),
                        "available_at": row.get("available_at", ""),
                        "published_at": row.get("published_at", ""),
                        "decision_date": _safe_date(row.get("available_at", "")),
                        "matched_tickers": _matched_tickers(row),
                        "source": row.get("source", ""),
                        "source_type": row.get("source_type", ""),
                        "source_family": row.get("source_family", ""),
                        "event_type": row.get("event_type", ""),
                        "split": row.get("split", ""),
                        "extractor_model": model_name,
                        "extractor_status": status,
                        "extractor_confidence": f"{confidence:.6f}",
                        "evidence_summary": evidence_summary,
                    }
                    for spec in schema["features"]:
                        name = spec["name"]
                        out[name] = f"{values[name]:.6f}"
                    writer.writerow(out)
                    handle.flush()
                    written += 1
                    new_rows += 1
                    if (
                        args.max_consecutive_failures > 0
                        and consecutive_failures >= args.max_consecutive_failures
                    ):
                        stop_reason = f"max_consecutive_failures:{consecutive_failures}"
                        print(
                            f"Stopping after {consecutive_failures} consecutive failures. "
                            f"Completed rows are preserved in {args.output}. "
                            "Fix the provider/API issue and rerun without --restart.",
                            file=sys.stderr,
                        )
                        break
                    if not args.dry_run and args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                if args.limit and written >= args.limit:
                    break

    manifest = {
        "status": "completed" if not stop_reason else "stopped",
        "stop_reason": stop_reason,
        "dry_run": args.dry_run,
        "schema": portable_path(args.schema),
        "feature_set": schema.get("feature_set", ""),
        "feature_columns": [item["name"] for item in schema["features"]],
        "input_files": len(files),
        "output": portable_path(args.output),
        "rows_written": written,
        "new_rows": new_rows,
        "failed_rows": failed_rows,
        "model": "dry_run_rule_like_stub" if args.dry_run else args.model,
        "base_url_host": re.sub(r"^https?://", "", _chat_completions_url(args.base_url)).split("/")[0],
        "elapsed_seconds": round(time.time() - started_at, 3),
        "resume": args.resume and not args.restart,
        "api_key_was_not_written": True,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
