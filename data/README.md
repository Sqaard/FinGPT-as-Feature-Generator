# Data

This folder contains the toy text/data handoff for the first PPO text-feature
loop.

## Text documents

- `train_2010_2021/`: documents with `split=train`, aligned with the training
  period before 2021-10-01.
- `test_2021_2023/`: documents with `split=test`, aligned with the OOS period
  from 2021-10-01 to 2023-03-01.

Each JSONL row is one document/retrieval unit with `available_at`,
`published_at`, `matched_tickers`, `source_type`, `title`, and `body`.

## Base price panel

- `processed_final_fixed_external_lagclean_full.csv`

This is the daily price/fundamental/macro panel used for merge. The merge script
aggregates document features by date/ticker and fills missing text features with
zero.

