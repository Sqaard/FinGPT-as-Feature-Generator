# DRL Issue #7: Document and source quality

Goal: identify which document families produce usable text signals before PPO consumes them.

## Source-family quality

| source_family | documents | quality_mean | signal_density_mean | timestamp_integrity_mean | next_day_abs_return_mean | noise_flag | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| company_earnings_release | 512 | 0.7868 | 0.304 | 1 | 0.0224 | none | keep_or_upweight |
| sec_exhibit | 655 | 0.7672 | 0.2098 | 1 | 0.01772 | none | keep_or_upweight |
| official_macro | 18240 | 0.7527 | 0.1806 | 1 | nan | low_text_signal_density | keep_but_downweight_for_directional_signal |
| company_ir | 598 | 0.7427 | 0.2424 | 1 | 0.01577 | none | keep_or_upweight |

## Scoring formula

`quality = 0.30 relevance + 0.25 signal_density + 0.20 timestamp_integrity + 0.15 source_reliability + 0.10 extraction_ok`

The next-day return column is an event-style diagnostic, not an alpha claim.
PPO impact should be added after the ablation matrix from Issue #6 has trained runs.

```json
{
  "status": "completed",
  "doc_dirs": [
    "data/train_2010_2021",
    "data/test_2021_2023"
  ],
  "features": "artifacts/text_features_mistral.csv",
  "panel": "artifacts/processed_final_fixed_external_lagclean_full_WITH_TEXT_MISTRAL.csv",
  "document_quality": "artifacts/document_source_quality/document_quality.csv",
  "source_family_quality": "artifacts/document_source_quality/source_family_quality.csv",
  "report": "reports/drl_document_source_quality.md",
  "documents_scored": 20005,
  "source_families": 4,
  "ppo_impact_note": "PPO impact requires trained ablations from Issue #6; this script adds event-style next-day return diagnostics now."
}
```