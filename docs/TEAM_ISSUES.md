# Proposed GitHub Issues

| Lane | Issue | Output |
|---|---|---|
| LLM | Compare LLM APIs on the same documents | quality table by model and source type |
| LLM | Design source-specific extraction prompts | prompts for SEC, earnings, macro, company IR |
| Features | Run feature-group ablations | PPO results by feature family |
| Features | Add temporal smoothing | 3d/5d/21d decay, rolling risk/sentiment |
| DRL | Normalize text features for PPO | train-only scalers and normalized panels |
| DRL | Test PPO-side text integration | reward/risk text use or two-branch policy |
| DRL | Score document/source quality | relevance, signal density, source reliability |

## Issue Bodies

### LLM: Compare LLM APIs on the same extraction set

Goal: make text-to-numerical extraction more accurate and reliable.

Tasks:
- compare Mistral with stronger paid/free LLM APIs on the same documents;
- keep the output schema fixed;
- log good and bad extraction cases;
- report quality by source type: SEC filings, earnings releases, macro, company IR.

Deliverable: `reports/llm_model_comparison.md` plus sample predictions.

### LLM: Source-specific prompts and schema improvements

Goal: improve extraction by adapting prompts to document type.

Tasks:
- suggest separate prompts for SEC sections, earnings releases, macro documents, and company IR;
- identify ambiguous features in the current schema;
- decide whether fine-tuning is realistic.

Deliverable: prompt files and a short schema improvement note.

### Features: Feature-group ablations

Goal: understand which text features help or hurt PPO.

Test groups:
- only risk / uncertainty;
- only sentiment / price impact;
- only forward-looking / earnings guidance;
- only macro financial conditions;
- only top-k features selected by event-study results or train-period correlation.

Deliverable: ablation table with return, Sharpe, max drawdown, and interpretation.

### Features: Temporal smoothing and decay

Goal: avoid one-day-only document effects.

Tasks:
- create 3-day, 5-day, and 21-day decayed text features;
- create rolling max risk;
- create rolling average sentiment;
- add event freshness decay.

Example: if a high-risk 10-K appears on Monday, the risk signal should decay
over several trading days instead of disappearing on Tuesday.

Deliverable: smoothed feature panel and comparison against unsmoothed features.

### DRL: Train-only normalization for text features

Goal: make text features safer for PPO.

Tasks:
- fit scalers only on the train window;
- test z-score, robust scaling, and clipping;
- keep test/OOS strictly out of scaler fitting.

Deliverable: normalized PPO panels and comparison report.

Implemented in the DRL branch:

- `scripts/05_normalize_text_features_train_only.py`;
- `artifacts/normalized_text_panels/`;
- `reports/drl_train_only_normalization.md`.

### DRL: PPO-side text integration

Goal: test whether text works better outside the raw state vector.

Tasks:
- try text signals in reward/risk constraints;
- prototype a two-branch policy: market branch + text branch;
- compare against simple concatenation.

Deliverable: PPO-side experiment report and implementation notes.

Implemented in the DRL branch:

- `scripts/06_build_ppo_text_integration_configs.py`;
- `scripts/03_train_backtest_ppo_with_text.py --text-integration-strategy ...`;
- `artifacts/ppo_text_integration_configs/`;
- `reports/drl_ppo_text_integration.md`.

### DRL: Document and source quality scoring

Goal: identify which documents/sources produce useful signals.

Tasks:
- score relevance, signal density, timestamp integrity, and source reliability;
- compare document quality against event-study diagnostics and PPO impact;
- mark noisy source families.

Deliverable: source/document quality table and recommendations.

Implemented in the DRL branch:

- `scripts/07_score_document_source_quality.py`;
- `artifacts/document_source_quality/`;
- `reports/drl_document_source_quality.md`.
