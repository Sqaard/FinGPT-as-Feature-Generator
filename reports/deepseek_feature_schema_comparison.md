# DeepSeek Feature Schema Comparison

Goal: compare new text feature schemas using one DeepSeek extractor and the same stratified document sample.

This is an extraction diagnostic, not a PPO performance claim.

## Schema ranking

| schema_name | feature_count | ok_documents | validity_rate | nonzero_share | avg_feature_std | low_variance_feature_count | diagnostic_utility_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1_baseline_10 | 10 | 32 | 1 | 0.4719 | 0.2325 | 0 | 0.9389 |
| ppo_state_compact_7 | 7 | 32 | 1 | 0.3705 | 0.1845 | 0 | 0.8457 |
| literature_top_12 | 12 | 32 | 1 | 0.3255 | 0.188 | 0 | 0.8199 |
| source_quality_8 | 8 | 32 | 1 | 0.8672 | 0.2663 | 0 | 0.791 |
| risk_control_8 | 8 | 32 | 1 | 0.1289 | 0.1071 | 2 | 0.6928 |
| investor_operator_12 | 12 | 32 | 1 | 0.125 | 0.1257 | 0 | 0.6603 |

## Source-family behavior

| schema_name | source_family | nonzero_share | mean_abs_value | avg_feature_std |
| --- | --- | --- | --- | --- |
| investor_operator_12 | company_earnings_release | 0.2396 | 0.1052 | 0.2158 |
| investor_operator_12 | company_ir | 0.01042 | 0.003125 | 0.03046 |
| investor_operator_12 | official_macro | 0.05208 | 0.009375 | 0.04389 |
| investor_operator_12 | sec_exhibit | 0.1979 | 0.05625 | 0.1456 |
| literature_top_12 | company_earnings_release | 0.4271 | 0.1911 | 0.2951 |
| literature_top_12 | company_ir | 0.2604 | 0.08281 | 0.2085 |
| literature_top_12 | official_macro | 0.1771 | 0.05938 | 0.1777 |
| literature_top_12 | sec_exhibit | 0.4375 | 0.1505 | 0.2531 |
| ppo_state_compact_7 | company_earnings_release | 0.4107 | 0.167 | 0.2813 |
| ppo_state_compact_7 | company_ir | 0.2143 | 0.06518 | 0.189 |
| ppo_state_compact_7 | official_macro | 0.3929 | 0.1286 | 0.2476 |
| ppo_state_compact_7 | sec_exhibit | 0.4643 | 0.1527 | 0.2467 |
| risk_control_8 | company_earnings_release | 0.1719 | 0.07031 | 0.1918 |
| risk_control_8 | company_ir | 0.125 | 0.08438 | 0.2508 |
| risk_control_8 | official_macro | 0.07812 | 0.025 | 0.1199 |
| risk_control_8 | sec_exhibit | 0.1406 | 0.06563 | 0.2218 |
| source_quality_8 | company_earnings_release | 0.9688 | 0.7063 | 0.3253 |
| source_quality_8 | company_ir | 0.8125 | 0.5664 | 0.4181 |
| source_quality_8 | official_macro | 0.75 | 0.5594 | 0.4335 |
| source_quality_8 | sec_exhibit | 0.9375 | 0.7348 | 0.3581 |

## Extraction errors

| schema_name | status | count | example_error |
| --- | --- | --- | --- |

## How to read this

- High validity means DeepSeek returned schema-valid JSON.
- Very low nonzero share means the schema is too sparse for a daily PPO panel.
- Very low feature std means the feature is nearly constant and likely useless.
- Source-family imbalance means the schema may only work for one document type.
- If every live request failed with 401, the API key, base URL, and model name do not belong to the same provider.
- The best next PPO candidate should be compact, non-constant, and not dominated by boilerplate.

```json
{
  "status": "completed",
  "dry_run": false,
  "model": "DeepSeek-V4-Flash",
  "base_url": "https://llmapi.paratera.com/v1/chat/completions",
  "json_mode": false,
  "schema_config": "feature_schema_deepseek_candidates.json",
  "output_dir": "artifacts/deepseek_feature_schema_comparison",
  "report": "reports/deepseek_feature_schema_comparison.md",
  "sample_docs": "artifacts/deepseek_feature_schema_comparison/sample_docs.jsonl",
  "sample_doc_count": 32,
  "schemas": [
    "literature_top_12",
    "v1_baseline_10",
    "investor_operator_12",
    "risk_control_8",
    "source_quality_8",
    "ppo_state_compact_7"
  ],
  "predictions": "artifacts/deepseek_feature_schema_comparison/sample_predictions.jsonl",
  "ok_predictions": 192,
  "error_predictions": 0,
  "schema_summary": "artifacts/deepseek_feature_schema_comparison/schema_summary.csv",
  "feature_summary": "artifacts/deepseek_feature_schema_comparison/feature_summary.csv",
  "source_family_summary": "artifacts/deepseek_feature_schema_comparison/source_family_summary.csv",
  "error_summary": "artifacts/deepseek_feature_schema_comparison/error_summary.csv",
  "errors": "artifacts/deepseek_feature_schema_comparison/extraction_errors.jsonl"
}
```