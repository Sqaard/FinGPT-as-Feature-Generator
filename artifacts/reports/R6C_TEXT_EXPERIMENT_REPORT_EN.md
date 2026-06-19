# R6c Baseline and Financial Text Feature Ablation Report

## Abstract

This report consolidates the R6c no-text baseline, the R6c model augmented with ten DeepSeek-V2 text features, and the R6c text-feature-group ablations available in the current repository. The central question is whether numerical features extracted from financial documents by a large language model can improve a PPO portfolio policy, and whether different semantic categories of text have different effects.

The earlier single-seed experiments showed mixed results: direct concatenation of all ten text features reduced four-fold mean validation return and Sharpe, while the forward-earnings group appeared strongest in one frozen 2022–2023 evaluation. A new paired Priority 0 screen now tests these findings on validation folds 2020 and 2021 with seeds 42, 123, and 2026, including zero-text and train-date-shuffled controls.

The 30-run screen does not confirm the earlier forward-earnings result. Raw forward-earnings features reduce mean paired Sharpe by 0.0185 and improve Sharpe in only 2 of 6 fold-seed pairs. Raw text10 has a small positive mean Sharpe delta of 0.0062, but improves only 3 of 6 pairs, worsens mean drawdown, and increases turnover. Neither candidate passes the predefined 75% paired-win stability gate. The current decision is therefore to retain no-text R6c and redesign text as confidence-gated, event-aware features before further PPO integration.

## 1. Research Questions

The experiments address four main questions:

1. Within the same R6c architecture, does adding all ten text features outperform the no-text baseline?
2. Is the effect of the full text feature set stable across walk-forward validation periods?
3. Do risk, sentiment, forward earnings, and macro text have different portfolio effects?
4. Are text features better used as return-prediction inputs or as risk-control and information-quality gates?

The older `custom_custom` PPO experiments provide a useful supporting counterexample. However, their action representation, execution mechanics, and text schema differ from R6c, so they should not be interpreted as a controlled architecture comparison. The main conclusions in this report are based on within-R6c comparisons.

## 2. R6c Strategy and Experiment Directory

### 2.1 Experiment directory

[`artifacts/r6c_stage0_1_text_baseline_20260530`](../r6c_stage0_1_text_baseline_20260530/) is a snapshot of the R6c text-baseline experiment assembled on May 30, 2026. It contains:

- a runnable copy of the Stage 0.1 R6c project;
- the DeepSeek-V2 compact text10 configuration;
- models and validation outputs from four walk-forward folds;
- frozen 2022–2023 out-of-sample evaluation;
- comparisons against the no-text R6c baseline;
- experiment manifests, model archives, and follow-up recommendations.

“Text baseline” in this directory means the simplest text integration method: concatenating ten fixed numerical text features with the PPO state. It does not mean the no-text model.

### 2.2 Meaning of R6c

R6c is an internal version name in the project's R-series strategy development. It is not a standard finance or reinforcement-learning acronym. Its full configuration name is:

```text
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

The name describes the execution design:

| Name component | Meaning |
| --- | --- |
| `root` | Decide the aggregate cash/risky-asset split before allocating among stocks |
| `K20` | Use an approximately 20-day execution window at the root risk-allocation level |
| `stock_K5` | Use an approximately 5-day execution window for stock-level adjustments |
| `PD` | Apply a proportional-derivative-like execution rule based on target-weight error and its change |
| `mild_slice` | Adjust positions gradually instead of executing the full target immediately |
| `group` | Discover stock groups from residual correlations and control group concentration |
| `riskaware` | Gate buying and rotation using risk stress, recovery conditions, and confidence |
| `top8_sell12` | Prioritize up to 8 buys and 12 sells per day |
| `rotation` | Permit stock-to-stock rotation under a constrained turnover budget |
| `v1` | First version of this execution configuration |

R6c is still trained with PPO, but it does not emit unconstrained trade impulses. It generates a constrained portfolio target and converts that target into executed weights through risk gates and gradual execution:

```text
market and stock state
    → PPO policy
    → cash/risky-asset root allocation
    → risky-asset stock weights
    → grouping, risk gates, and Top-K selection
    → gradually executed portfolio weights
```

The underlying configuration uses `root_split_beta_dirichlet` and `root_split_weights`, which are designed to produce non-negative portfolio weights with the required sum constraint. R6c inherits from R6b: R6 introduced Top-K stock rotation, R6b added residual-correlation stock groups, and R6c added risk-aware gates and revised the buy/sell counts.

See [`stage0_1_r6c_deepseek_v2_text.yaml`](../r6c_stage0_1_text_baseline_20260530/rl_stage0_1_r6c_project/configs/stage0_1_r6c_deepseek_v2_text.yaml) for the implementation contract.

## 3. Data, Text Features, and Evaluation Protocol

### 3.1 Data overview

The current panel contains:

- 29 stocks;
- 96,019 daily stock rows;
- dates from January 4, 2010 to February 28, 2023;
- 20,005 timestamped financial documents;
- 16,446 training-period documents and 3,559 test-period documents.

The source distribution is highly imbalanced:

| Document source | Count | Share |
| --- | ---: | ---: |
| Official macro documents | 18,240 | 91.18% |
| Core company IR documents | 844 | 4.22% |
| SEC filing exhibits | 655 | 3.27% |
| Company IR review documents | 266 | 1.33% |

The text panel therefore represents macro conditions much more heavily than company-specific earnings or event information. This structure may be more useful for market-state and risk control than for cross-sectional stock selection.

### 3.2 DeepSeek-V2 text10 features

The full text model uses:

| Feature | Meaning |
| --- | --- |
| `text_alpha_direction` | Net investable directional evidence |
| `text_downside_risk` | Downside-risk intensity |
| `text_uncertainty` | Conditional, uncertain, or unclear language |
| `text_macro_stress` | Macro or cross-asset stress |
| `text_earnings_pressure` | Earnings, revenue, margin, EPS, or guidance pressure |
| `text_balance_sheet_stress` | Leverage, liquidity, funding, or solvency stress |
| `text_signal_confidence` | Confidence that the signal is timely, specific, and credible |
| `text_evidence_specificity` | Specificity of document-grounded evidence |
| `text_numeric_evidence_density` | Density of useful financial and operating numerical evidence |
| `text_boilerplate_intensity` | Intensity of generic or legal boilerplate |

The complete contract is available in [`feature_schema_deepseek_v2_ppo_compact.json`](../../feature_schema_deepseek_v2_ppo_compact.json).

### 3.3 Training and validation

The original full four-fold experiment used:

- 350,000 PPO training timesteps;
- random seed 42;
- 0.1% transaction-cost rate;
- normalization statistics fitted only on each fold's training period;
- a reward that includes return, turnover, drawdown, concentration, and action-change terms.

The walk-forward design is:

| Fold | Training period | Validation period |
| --- | --- | --- |
| `fold_2018` | Jan. 4, 2010–Dec. 29, 2017 | 2018 |
| `fold_2019` | Jan. 4, 2010–Dec. 31, 2018 | 2019 |
| `fold_2020` | Jan. 4, 2010–Dec. 31, 2019 | 2020 |
| `fold_2021` | Jan. 4, 2010–Dec. 31, 2020 | 2021 |

The frozen out-of-sample period is January 3, 2022 through February 27, 2023. The main metrics are total return, Sharpe ratio, maximum drawdown, and mean L1 turnover.

The later Priority 0 confirmation screen used:

- validation folds `fold_2020` and `fold_2021`;
- random seeds 42, 123, and 2026;
- paired no-text and text runs with identical fold and seed;
- raw text10, raw forward-earnings, zero-forward-earnings, and train-date-shuffled forward-earnings variants;
- 30 completed runs in total.

Candidate selection in this confirmation screen did not use the frozen 2022–2023 period.

## 4. Full Text10 Baseline Experiment

### 4.1 Four-fold walk-forward results

| Metric | R6c no text | R6c + text10 | Text minus baseline |
| --- | ---: | ---: | ---: |
| Mean return | 10.05% | 9.10% | -0.95 pct |
| Mean Sharpe | 1.0939 | 1.0612 | -0.0326 |
| Mean maximum drawdown | -7.34% | -7.13% | +0.21 pct |
| Mean L1 turnover | 0.00844 | 0.00831 | -0.00013 |

The full text state slightly reduces mean return and Sharpe while slightly improving drawdown and turnover. Across validation periods, text10 therefore behaves more like a small defensive adjustment than a stable source of incremental alpha.

The fold-level results for R6c + text10 are:

| Validation year | Return | Sharpe | Maximum drawdown | Mean L1 turnover |
| --- | ---: | ---: | ---: | ---: |
| 2018 | -0.75% | -0.0069 | -13.29% | 0.00927 |
| 2019 | 13.02% | 1.7162 | -4.22% | 0.00778 |
| 2020 | 16.54% | 1.3784 | -6.52% | 0.00789 |
| 2021 | 7.59% | 1.1573 | -4.49% | 0.00832 |

The large differences across years—especially the weak 2018 result—indicate substantial period dependence.

### 4.2 Frozen out-of-sample results

| Strategy | Return | Sharpe | Maximum drawdown | Mean L1 turnover |
| --- | ---: | ---: | ---: | ---: |
| R6c no text | -1.92% | -0.1015 | -11.78% | 0.00744 |
| R6c + text10 | -1.61% | -0.0640 | -12.04% | 0.00762 |
| Text minus baseline | +0.31 pct | +0.0374 | -0.27 pct | +0.00018 |

The full text model slightly improves frozen out-of-sample return and Sharpe. However, both strategies remain loss-making, and the text model has slightly worse drawdown and higher turnover.

### 4.3 Baseline-experiment conclusion

The walk-forward and frozen out-of-sample evidence is not fully consistent:

- full text10 is slightly worse on validation return and Sharpe;
- full text10 is slightly better on return and Sharpe in one frozen period;
- the frozen improvement is small and comes with worse drawdown and turnover.

Full text10 should therefore be treated as a weak positive out-of-sample signal, not as a stable improvement. The disagreement may reflect market-regime dependence, random-seed variation, text sparsity, or model-selection noise.

## 5. Text Feature Group Ablation

The ablation experiments retain the same R6c architecture, train separate text subsets on `fold_2021`, and evaluate them over the same frozen 2022–2023 out-of-sample period.

### 5.1 Feature groups

| Group | Active features |
| --- | --- |
| Risk/uncertainty | `text_downside_risk`, `text_uncertainty`, `text_balance_sheet_stress`, `text_boilerplate_intensity` |
| Sentiment/price impact | `text_alpha_direction`, `text_signal_confidence`, `text_evidence_specificity` |
| Forward earnings/evidence | `text_earnings_pressure`, `text_numeric_evidence_density`, `text_signal_confidence` |
| Macro financial conditions | `text_macro_stress`, `text_uncertainty` |
| Top-3 train correlation | `text_numeric_evidence_density`, `text_boilerplate_intensity`, `text_evidence_specificity` |

The groups are not fully disjoint. For example, `text_signal_confidence` appears in both the sentiment and forward-earnings groups, while `text_uncertainty` appears in both the risk and macro groups. The experiment therefore measures combinations rather than unique single-feature contributions.

### 5.2 Frozen out-of-sample results

| Strategy | Return | Sharpe | Maximum drawdown | Mean L1 turnover |
| --- | ---: | ---: | ---: | ---: |
| R6c no text | -1.92% | -0.1015 | -11.78% | 0.00744 |
| All text10 | -1.61% | -0.0640 | -12.04% | 0.00762 |
| Risk/uncertainty | -1.43% | -0.0539 | -12.15% | 0.00722 |
| Sentiment/price impact | -1.81% | -0.0588 | -13.57% | 0.00806 |
| Forward earnings/evidence | **-0.65%** | **0.0105** | -12.37% | 0.00770 |
| Macro financial conditions | -1.62% | -0.0741 | **-11.59%** | 0.00735 |
| Top-3 train correlation | -1.37% | -0.0349 | -12.92% | 0.00789 |

Changes relative to the no-text baseline are:

| Feature group | Return change | Sharpe change | Maximum-drawdown change |
| --- | ---: | ---: | ---: |
| All text10 | +0.31 pct | +0.0374 | -0.27 pct |
| Risk/uncertainty | +0.49 pct | +0.0476 | -0.37 pct |
| Sentiment/price impact | +0.11 pct | +0.0427 | -1.79 pct |
| Forward earnings/evidence | **+1.27 pct** | **+0.1120** | -0.59 pct |
| Macro financial conditions | +0.30 pct | +0.0274 | **+0.19 pct** |
| Top-3 train correlation | +0.55 pct | +0.0666 | -1.15 pct |

A maximum drawdown closer to zero is better. A negative drawdown change in the table therefore indicates deterioration.

## 6. Interpretation of the Ablations

### 6.1 Forward earnings were the strongest single-run screening signal

The forward-earnings group reduces the cumulative loss from -1.92% to -0.65% and raises Sharpe from -0.1015 to 0.0105. It is the only text group with a positive Sharpe.

This historical result supported the following hypothesis:

> Company-specific prospective information is more useful when it is accompanied by numerical evidence and a high-confidence quality signal.

The later multi-seed confirmation screen does not validate this hypothesis for the raw feature group. The single frozen-period result should therefore be treated as candidate generation, not evidence that forward-earnings text reliably improves PPO.

### 6.2 Macro text behaves more like a risk-control signal

The macro group produces only a small return and Sharpe improvement, but it has a better maximum drawdown and lower turnover than the no-text baseline. Macro information is broad, shared across stocks, and relatively slow-moving, so it is not naturally suited to cross-sectional stock ranking.

Macro text may be more appropriate for:

- adjusting the cash/risky-asset split;
- changing the portfolio risk budget;
- controlling rebalancing speed;
- gating exposure under stressed regimes.

A better architecture would route macro text only to the R6c root risk decision instead of repeating the same signal in every stock observation.

### 6.3 Generic sentiment is the least convincing group

The sentiment group improves return by only 0.11 percentage points while producing the worst drawdown and highest turnover. Possible explanations include:

- duplication of price momentum and volatility information already in the state;
- confusion between historical negative descriptions and future direction;
- information being priced before the model acts;
- weak directional signals triggering unnecessary reallocations.

These explanations are consistent with the observed metrics but have not yet been verified through trade-level causal diagnostics.

### 6.4 Risk and uncertainty provide weak incremental value

The risk group outperforms full text10 on return and Sharpe and reduces average turnover, but its maximum drawdown is slightly worse than the baseline. Risk text therefore does not behave as a consistent downside-protection mechanism. It may instead improve allocation in selected states.

### 6.5 Univariate correlation is only a screening heuristic

The Top-3 correlation group outperforms full text10, but the largest absolute train-period Pearson correlation is only 0.0267. Individual text features have very weak linear relationships with next-day returns.

Text value is therefore more likely to arise from:

- interactions among text features;
- conditional relationships with market state;
- effects on ranking, allocation, or risk control;
- filtering low-quality text rather than directly forecasting return.

Univariate correlation can reduce the candidate set, but it should not be treated as evidence of stable feature importance.

### 6.6 Compact semantic groups may be better than full concatenation

Several compact groups outperform the full text10 state on return. Ten weak and partially redundant dimensions can increase estimation and credit-assignment difficulty for PPO, encouraging spurious interactions or feature interference.

The historical single-seed evidence suggested:

> Small text groups with explicit economic roles may be preferable to unrestricted concatenation, but they still require paired multi-seed confirmation.

## 7. Priority 0 Paired Multi-Seed Confirmation

### 7.1 Design

The confirmation screen compared five variants:

| Variant | Purpose |
| --- | --- |
| R6c base | Paired no-text reference |
| Raw forward earnings | Test the strongest historical group |
| Shuffled forward earnings | Test whether correct temporal alignment matters |
| Zero forward earnings | Control for observation shape and training variance |
| Raw text10 | Re-evaluate unrestricted state concatenation |

Each variant was trained on validation folds 2020 and 2021 with seeds 42, 123, and 2026. This gives six paired observations per variant and 30 completed runs.

### 7.2 Aggregate results

| Variant | Mean return | Mean Sharpe | Sharpe delta vs base | Positive paired Sharpe | Mean max DD | Mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R6c base | 12.30% | 1.2539 | 0.0000 | — | -5.56% | 0.00786 |
| Forward earnings raw | 11.82% | 1.2354 | -0.0185 | 2/6 | -5.43% | 0.00800 |
| Forward earnings shuffled | 12.23% | 1.2510 | -0.0029 | 3/6 | -5.53% | 0.00817 |
| Forward earnings zero | 12.39% | 1.2523 | -0.0016 | 2/6 | -5.59% | 0.00812 |
| Raw text10 | 12.67% | 1.2601 | +0.0062 | 3/6 | -5.81% | 0.00828 |

### 7.3 Promotion-gate decision

The predefined gate requires a positive mean paired validation Sharpe delta, positive Sharpe delta in at least 75% of fold-seed pairs, controlled drawdown and turnover, and superiority to zero and shuffled controls.

- Raw forward earnings fails the mean-Sharpe gate and improves only 33.3% of pairs. It does not beat either negative control.
- Raw text10 has a small positive mean Sharpe delta, but improves only 50% of pairs, worsens mean maximum drawdown by 0.26 percentage points, and raises turnover by about 5.4%.

Neither candidate is promoted to the full four-fold stage. The no-text R6c policy remains the reference.

### 7.4 Seed instability

Raw text10 paired Sharpe deltas range from -0.0648 to +0.1117. Forward-earnings deltas range from -0.1150 to +0.1072. Sign reversals across seeds on the same validation fold show that the observed text effects are comparable to PPO training variance.

## 8. Conclusions Supported by the Current Evidence

### 8.1 Relatively well supported

1. Direct concatenation of all ten text features does not consistently improve R6c across time.
2. The earlier forward-earnings frozen-OOS improvement does not replicate across paired seeds and validation folds.
3. Raw text10 produces a small positive average effect but not a stable one.
4. Zero and shuffled controls perform similarly to raw forward earnings, so observation dimensionality and PPO variance explain much of the apparent effect.
5. Current raw text features should not replace the no-text R6c policy.

### 8.2 Preliminary but unconfirmed

1. R6c's constrained hierarchical execution may help it use weak text signals.
2. Event-aware earnings features may work better than sparse raw levels.
3. Text effectiveness may depend on market period or risk regime.
4. Confidence and boilerplate intensity may be useful as explicit text-quality gates.

### 8.3 Claims not supported at present

1. Text features significantly improve absolute profitability.
2. R6c + text10 consistently outperforms the no-text R6c baseline.
3. `text_earnings_pressure` alone causes the forward-group improvement.
4. DeepSeek-V2 is necessarily better than Mistral.
5. R6c is definitively better than the older PPO at using text, because the experiments are not fully controlled.
6. The current differences are statistically significant or generalize to other markets and periods.

## 9. Limitations

The main limitations are:

- the confirmation screen covers three seeds, but only validation years 2020 and 2021;
- the older group ablations still train only on `fold_2021`;
- frozen evaluation covers only one 2022–2023 period;
- overlapping groups prevent unique feature attribution;
- no bootstrap confidence intervals or significance tests are reported;
- 91.18% of documents are macro sources, leaving limited company-event coverage;
- `text_earnings_pressure` is nonzero in only about 1.9% of panel rows;
- text is mainly integrated through state concatenation rather than a separate encoder or root-level risk gate;
- the text schemas differ across LLM experiments, preventing clean attribution to the extractor model itself.

## 10. Recommended Follow-up Experiments

The highest-priority next steps are:

1. Replace raw earnings levels with an event mask, event age, and causal 1/3/5/10-day decay.
2. Gate directional earnings signals using confidence, specificity, numerical evidence, and `1 - boilerplate`.
3. Distinguish missing text, neutral text, and stale text.
4. Repeat the same paired two-fold, three-seed screen against zero and shuffled controls.
5. Only after a redesigned feature set passes the gate, test a zero-initialized residual text adapter.
6. Route macro text only to the root cash/risk allocation decision.
7. Report results by volatility regime, drawdown state, earnings season, source type, event age, and signal sign.
8. Increase the share of company releases, earnings documents, and SEC filings to reduce macro-source dominance.

## 11. Overall Conclusion

The current experiments do not establish that adding LLM-derived text features generally improves PPO trading performance. A more accurate conclusion is:

> The current raw text representations do not provide a stable improvement over no-text R6c. The earlier forward-earnings result does not survive paired multi-seed validation, while raw text10 has only a small and inconsistent average benefit. Future work should redesign text as sparse, event-aware, quality-gated signals and require superiority to zero and shuffled controls before changing the PPO architecture.

The completed Priority 0 screen supports retaining the no-text R6c policy. No current text candidate qualifies for frozen-OOS promotion.

## 12. Result File Index

- [R6c text results and follow-up recommendations](../r6c_stage0_1_text_baseline_20260530/POST_TRAINING_NEXT_STEPS.md)
- [R6c text launch and result record](../r6c_stage0_1_text_baseline_20260530/R6C_TEXT_LAUNCH_PREP.md)
- [Full text10 versus baseline frozen OOS comparison](../r6c_stage0_1_text_baseline_20260530/r6c_text_frozen_vs_baseline_fold2021.csv)
- [Cross-model text-effect discussion](../r6c_stage0_1_text_baseline_20260530/CROSS_MODEL_TEXT_EFFECT_COMPARISON.md)
- [R6c feature-group ablation summary](../r6c_text_feature_group_ablation/results/r6c_ablation_summary.csv)
- [Feature-group definitions](../r6c_text_feature_group_ablation/results/feature_groups.csv)
- [Train-period text-feature correlations](../r6c_text_feature_group_ablation/results/train_text_feature_correlations.csv)
- [Priority 0 paired-screen report](../text_improvement_plan/priority0_report.md)
- [Priority 0 aggregate results](../text_improvement_plan/text_experiment_summary.csv)
- [Priority 0 paired fold-seed results](../text_improvement_plan/text_experiment_paired.csv)
- [Existing consolidated experiment report](../FINAL_EXPERIMENT_REPORT.md)
