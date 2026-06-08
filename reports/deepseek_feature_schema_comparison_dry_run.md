# DeepSeek Feature Schema Comparison

Goal: compare new text feature schemas using one DeepSeek extractor and the same stratified document sample.

This is an extraction diagnostic, not a PPO performance claim.

## Schema ranking

| schema_name | feature_count | ok_documents | validity_rate | nonzero_share | avg_feature_std | low_variance_feature_count | diagnostic_utility_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literature_top_12 | 12 | 8 | 1 | 0.5208 | 0.187 | 4 | 0.8387 |

## Source-family behavior

| schema_name | source_family | nonzero_share | mean_abs_value | avg_feature_std |
| --- | --- | --- | --- | --- |
| literature_top_12 | company_earnings_release | 0.4583 | 0.2083 | 0.3012 |
| literature_top_12 | company_ir | 0.375 | 0.1979 | 0.3714 |
| literature_top_12 | official_macro | 0.6667 | 0.2257 | 0.3377 |
| literature_top_12 | sec_exhibit | 0.5833 | 0.3772 | 0.4887 |

## How to read this

- High validity means DeepSeek returned schema-valid JSON.
- Very low nonzero share means the schema is too sparse for a daily PPO panel.
- Very low feature std means the feature is nearly constant and likely useless.
- Source-family imbalance means the schema may only work for one document type.
- The best next PPO candidate should be compact, non-constant, and not dominated by boilerplate.

```json
{
  "status": "completed",
  "dry_run": true,
  "model": "dry_run_keyword_stub",
  "base_url": "none",
  "schema_config": "feature_schema_deepseek_candidates.json",
  "output_dir": "artifacts/deepseek_feature_schema_comparison_dry_run",
  "report": "reports/deepseek_feature_schema_comparison_dry_run.md",
  "sample_docs": "artifacts/deepseek_feature_schema_comparison_dry_run/sample_docs.jsonl",
  "sample_doc_count": 8,
  "schemas": [
    "literature_top_12"
  ],
  "predictions": "artifacts/deepseek_feature_schema_comparison_dry_run/sample_predictions.jsonl",
  "ok_predictions": 8,
  "error_predictions": 0,
  "schema_summary": "artifacts/deepseek_feature_schema_comparison_dry_run/schema_summary.csv",
  "feature_summary": "artifacts/deepseek_feature_schema_comparison_dry_run/feature_summary.csv",
  "source_family_summary": "artifacts/deepseek_feature_schema_comparison_dry_run/source_family_summary.csv",
  "errors": "artifacts/deepseek_feature_schema_comparison_dry_run/extraction_errors.jsonl"
}
```