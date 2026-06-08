# DeepSeek Feature Schema Comparison

Goal: compare new text feature schemas using one DeepSeek extractor and the same stratified document sample.

This is an extraction diagnostic, not a PPO performance claim.

## Schema ranking

| schema_name | feature_count | ok_documents | validity_rate | nonzero_share | avg_feature_std | low_variance_feature_count | diagnostic_utility_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literature_top_12 | 12 | 4 | 1 | 0.4792 | 0.1108 | 4 | 0.7621 |

## Source-family behavior

| schema_name | source_family | nonzero_share | mean_abs_value | avg_feature_std |
| --- | --- | --- | --- | --- |
| literature_top_12 | company_earnings_release | 0.6667 | 0.2917 | 0.299 |
| literature_top_12 | company_ir | 0.08333 | 0.08333 | 0.2764 |
| literature_top_12 | official_macro | 0.6667 | 0.2319 | 0.3524 |
| literature_top_12 | sec_exhibit | 0.5 | 0.1875 | 0.2724 |

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
  "dry_run": true,
  "model": "dry_run_keyword_stub",
  "base_url": "none",
  "json_mode": false,
  "schema_config": "feature_schema_deepseek_candidates.json",
  "output_dir": "artifacts/deepseek_feature_schema_comparison_dry_run2",
  "report": "reports/deepseek_feature_schema_comparison_dry_run2.md",
  "sample_docs": "artifacts/deepseek_feature_schema_comparison_dry_run2/sample_docs.jsonl",
  "sample_doc_count": 4,
  "schemas": [
    "literature_top_12"
  ],
  "predictions": "artifacts/deepseek_feature_schema_comparison_dry_run2/sample_predictions.jsonl",
  "ok_predictions": 4,
  "error_predictions": 0,
  "schema_summary": "artifacts/deepseek_feature_schema_comparison_dry_run2/schema_summary.csv",
  "feature_summary": "artifacts/deepseek_feature_schema_comparison_dry_run2/feature_summary.csv",
  "source_family_summary": "artifacts/deepseek_feature_schema_comparison_dry_run2/source_family_summary.csv",
  "error_summary": "artifacts/deepseek_feature_schema_comparison_dry_run2/error_summary.csv",
  "errors": "artifacts/deepseek_feature_schema_comparison_dry_run2/extraction_errors.jsonl"
}
```