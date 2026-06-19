# Priority 0 PPO Text Screen Report

## Scope

The screen ran five variants on `fold_2020` and `fold_2021` with seeds `42`,
`123`, and `2026`. Every text result is paired with a no-text R6c run using the
same fold and seed.

Completed jobs: `30 / 30`.

## Aggregate results

| Variant | Mean return | Mean Sharpe | Sharpe delta vs base | Positive paired Sharpe | Mean max DD | Mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R6c base | 12.30% | 1.2539 | 0.0000 | — | -5.56% | 0.00786 |
| Forward earnings raw | 11.82% | 1.2354 | -0.0185 | 2/6 | -5.43% | 0.00800 |
| Forward earnings shuffled | 12.23% | 1.2510 | -0.0029 | 3/6 | -5.53% | 0.00817 |
| Forward earnings zero | 12.39% | 1.2523 | -0.0016 | 2/6 | -5.59% | 0.00812 |
| Raw text10 | 12.67% | 1.2601 | +0.0062 | 3/6 | -5.81% | 0.00828 |

## Gate evaluation

The promotion rules in `plan.md` require a positive mean paired validation
Sharpe delta, positive Sharpe delta in at least 75% of fold-seed pairs, no
material drawdown degradation, acceptable turnover, and superiority to the
negative controls.

### Forward earnings

- Mean paired return delta: `-0.48` percentage points.
- Mean paired Sharpe delta: `-0.0185`.
- Positive paired Sharpe results: `2/6`, or `33.3%`.
- It does not beat the zero or shuffled controls.

Decision: **reject the raw forward-earnings feature group**. It fails the
primary mean-Sharpe and stability gates.

### Raw text10

- Mean paired return delta: `+0.38` percentage points.
- Mean paired Sharpe delta: `+0.0062`.
- Positive paired Sharpe results: `3/6`, or `50.0%`.
- Mean maximum drawdown worsened by `0.26` percentage points.
- Mean turnover increased by approximately `5.4%`.

Decision: **do not promote raw text10**. Its mean result is slightly positive,
but the effect is unstable and fails the required `75%` paired-win threshold.

## Fold-seed instability

Raw text10 Sharpe deltas ranged from:

- `-0.0648` for seed 42, fold 2021;
- to `+0.1117` for seed 2026, fold 2021.

Forward-earnings Sharpe deltas ranged from:

- `-0.1150` for seed 2026, fold 2020;
- to `+0.1072` for seed 123, fold 2021.

The sign changes across seeds on the same fold show that the observed text
effects are comparable to PPO training variance rather than a stable feature
advantage.

## Conclusion

Priority 0 does not justify a full four-fold promotion run for either current
text representation. Keep the no-text R6c policy as the reference.

The next justified work is data/feature redesign:

1. Build confidence-gated, event-aware earnings features.
2. Distinguish missing, neutral, and stale text explicitly.
3. Add event age and causal 1/3/5/10-day decay.
4. Repeat this same paired two-fold, three-seed screen.
5. Implement the residual adapter only after a redesigned feature set beats
   both zero and shuffled controls.

The frozen 2022-2023 OOS period was not used for selection.

## Result files

- `text_experiment_runs.csv`
- `text_experiment_paired.csv`
- `text_experiment_summary.csv`
