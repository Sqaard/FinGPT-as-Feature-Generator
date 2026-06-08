# DeepSeek Feature Schema Comparison

Goal: compare new text feature schemas using one DeepSeek extractor and the same stratified document sample.

This is an extraction diagnostic, not a PPO performance claim.

## Schema ranking

| schema_name | feature_count | ok_documents | validity_rate | nonzero_share | avg_feature_std | low_variance_feature_count | diagnostic_utility_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_state_compact_7 | 7 | 0 | 0 | 0 | 0 | 7 | 0 |

## Source-family behavior

| schema_name | source_family | nonzero_share | mean_abs_value | avg_feature_std |
| --- | --- | --- | --- | --- |

## Extraction errors

| schema_name | status | count | example_error |
| --- | --- | --- | --- |
| ppo_state_compact_7 | failed:RuntimeError | 1 | HTTP 401 Unauthorized: {"error":{"message":"Authentication Error,","type":"auth_error","param":"None","code":"401"}} |

## How to read this

- High validity means DeepSeek returned schema-valid JSON.
- Very low nonzero share means the schema is too sparse for a daily PPO panel.
- Very low feature std means the feature is nearly constant and likely useless.
- Source-family imbalance means the schema may only work for one document type.
- If every live request failed with 401, the API key, base URL, and model name do not belong to the same provider.
- The best next PPO candidate should be compact, non-constant, and not dominated by boilerplate.

```json
{
  "status": "failed_no_successful_predictions",
  "dry_run": false,
  "model": "DeepSeek-V4-Flash",
  "base_url": "https://llmapi.paratera.com/v1/chat/completions",
  "json_mode": false,
  "schema_config": "feature_schema_deepseek_candidates.json",
  "output_dir": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke",
  "report": "reports/deepseek_feature_schema_comparison_auth_failure_smoke.md",
  "sample_docs": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/sample_docs.jsonl",
  "sample_doc_count": 1,
  "schemas": [
    "ppo_state_compact_7"
  ],
  "predictions": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/sample_predictions.jsonl",
  "ok_predictions": 0,
  "error_predictions": 1,
  "schema_summary": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/schema_summary.csv",
  "feature_summary": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/feature_summary.csv",
  "source_family_summary": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/source_family_summary.csv",
  "error_summary": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/error_summary.csv",
  "errors": "artifacts/deepseek_feature_schema_comparison_auth_failure_smoke/extraction_errors.jsonl"
}
```