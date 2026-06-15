# Final Experiment Report: Text Feature Groups for CHRL

## Objective

The experiment asked which text feature groups help or hurt PPO:

- risk / uncertainty;
- sentiment / price impact;
- forward-looking / earnings guidance;
- macro financial conditions;
- top-k features selected from train-period correlation.

The primary completed ablation is the CHRL frozen out-of-sample evaluation on
`fold_2021`. Older `custom_custom` PPO results are included as supporting
evidence, but they are not directly comparable to CHRL because the policy
architecture, text schema, and execution mechanics differ.

## Main CHRL Ablation Results

| Strategy | Return | Sharpe | Max drawdown | Return delta vs. baseline | Sharpe delta vs. baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| CHRL model | -1.92% | -0.1015 | -11.78% | 0.00% | 0.0000 |
| CHRL model + all raw text10 | -1.61% | -0.0640 | -12.04% | +0.31% | +0.0374 |
| Only risk / uncertainty | -1.43% | -0.0539 | -12.15% | +0.49% | +0.0476 |
| Only sentiment / price impact | -1.81% | -0.0588 | -13.57% | +0.11% | +0.0427 |
| Only forward-looking / earnings guidance | **-0.65%** | **0.0105** | -12.37% | **+1.27%** | **+0.1120** |
| Only macro financial conditions | -1.62% | -0.0741 | **-11.59%** | +0.30% | +0.0274 |
| Top-3 train-correlation features | -1.37% | -0.0349 | -12.92% | +0.55% | +0.0666 |

All CHRL text groups improved return and Sharpe relative to the no-text CHRL
baseline. However, only the forward/earnings group produced a positive Sharpe,
and every strategy still had a negative cumulative return.

## Feature-Group Interpretation

### Most useful: forward-looking / earnings guidance

This was the strongest group by a clear margin. It reduced the loss from
`-1.92%` to `-0.65%` and raised Sharpe from `-0.1015` to `0.0105`. Its active
features were:

- `text_earnings_pressure`
- `text_numeric_evidence_density`
- `text_signal_confidence`

The result suggests that text is most useful when it describes prospective,
company-specific information and is supported by quantitative evidence or
confidence. This is the best candidate for confirmation with additional seeds
and folds.

### Useful but mixed: risk / uncertainty

Risk and uncertainty produced the second-best grouped result by return and
improved Sharpe materially. Its drawdown was slightly worse than baseline,
though, so the gain appears to come from better return generation rather than
better downside control.

### Defensive but weak: macro financial conditions

Macro features generated only a small return and Sharpe improvement, but had
the best max drawdown (`-11.59%`). These features may be more suitable as a
risk or regime gate than as direct alpha inputs.

### Noisy or inefficient: sentiment / price impact

Sentiment/price features barely improved return and produced the worst
drawdown (`-13.57%`). Although Sharpe improved relative to baseline, the weak
return benefit and larger drawdown make this group the least convincing CHRL
addition. Generic directional sentiment appears noisier than
earnings/evidence-oriented information.

### Useful screening signal, not a stable selection rule: top-k correlation

The top-3 group outperformed the full text10 set on return and Sharpe, showing
that a smaller feature set can be preferable. However, the underlying
train-period correlations were extremely small: the largest absolute
correlation was only `0.0267`. The group's max drawdown also worsened to
`-12.92%`. Therefore, univariate train correlation should be treated as a
screening heuristic, not strong evidence of predictive importance.

## Architecture Dependence

The older flat `custom_custom` PPO provides an important counterexample:

| custom_custom strategy | Test return | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| PPO without text | -7.29% | -0.2140 | -20.20% |
| PPO with all text | -19.60% | -0.9152 | -22.28% |
| Sentiment / price impact | -14.21% | -0.6307 | -19.85% |
| Forward-looking / earnings guidance | -16.29% | -0.7941 | -23.53% |
| Macro financial conditions | -18.84% | -0.9719 | -21.57% |
| Top-k train-correlation | -14.19% | -0.6256 | -23.70% |

Every available text subset underperformed the no-text `custom_custom`
baseline on return and Sharpe. The risk/uncertainty run is absent from the
available logs, so that older ablation is incomplete.

This contrast indicates that text features are architecture-dependent. Raw
state concatenation overloads or distracts the older flat PPO, whereas the R6c
hierarchy can extract a small benefit from compact, semantically aligned text
features.

## Why the Results Likely Occurred

The experiment does not identify causal mechanisms directly, so the
explanations below are hypotheses supported by the observed metrics and model
design. They should be tested with targeted follow-up ablations.

### 1. Specific information is more useful than broad tone

The forward/earnings group contains `text_numeric_evidence_density` and
`text_signal_confidence` alongside `text_earnings_pressure`. This combination
can distinguish a concrete, supported earnings signal from vague language.
That distinction is economically relevant because forward guidance can affect
expected cash flows and analyst revisions beyond the filing date.

By contrast, `text_alpha_direction` in the sentiment group has a negative
train correlation with next-day return (`-0.0046`), and generic sentiment may
duplicate price momentum, analyst expectations, or market variables already
present in the state. It can also confuse negative tone with negative future
return: bad language may already be priced in, may produce a relief reaction,
or may describe known historical performance rather than new information.

This helps explain why the forward/earnings group generated the largest return
improvement while sentiment/price impact generated only `+0.11%`.

### 2. The useful signal is probably conditional and nonlinear

The individual linear correlations are all weak. Even the largest absolute
train correlation is only `0.0267`, while `text_earnings_pressure` itself is
only `0.0033`. The strong relative result of the forward/earnings group
therefore cannot be explained by a large standalone linear earnings signal.

A more plausible mechanism is interaction:

- earnings pressure matters more when evidence density is high;
- confidence can suppress ambiguous observations;
- a text signal may matter only under a compatible volatility, valuation, or
  market regime;
- text may change stock ranking or allocation without predicting the
  unconditional next-day return.

PPO can learn such state-dependent interactions, whereas the top-k procedure
ranks features using only one-feature Pearson correlation. This is also why
correlation ranking selected `text_boilerplate_intensity` but did not select
`text_earnings_pressure`.

### 3. Boilerplate may work as a quality filter, not an alpha signal

`text_boilerplate_intensity` has the second-largest absolute train correlation
and a negative sign (`-0.0237`). A plausible interpretation is that repetitive
or generic disclosures contain less incremental information. The policy may
use this feature to discount other text rather than to take a direct bearish
position.

This would explain why boilerplate appears in the useful top-k and
risk/uncertainty groups even though it is not naturally a directional return
forecast. It also argues for implementing boilerplate and confidence as
explicit reliability gates.

### 4. Turnover helps explain the drawdown differences

| CHRL strategy | Mean L1 turnover | Change vs. baseline | Max drawdown |
| --- | ---: | ---: | ---: |
| Baseline | 0.00744 | 0.0% | -11.78% |
| All text10 | 0.00762 | +2.4% | -12.04% |
| Risk / uncertainty | 0.00722 | -3.0% | -12.15% |
| Sentiment / price impact | 0.00806 | +8.3% | -13.57% |
| Forward / earnings | 0.00770 | +3.4% | -12.37% |
| Macro conditions | 0.00735 | -1.3% | -11.59% |
| Top-3 correlation | 0.00789 | +6.0% | -12.92% |

The two largest turnover increases occurred for sentiment and top-k, which
also had materially worse drawdowns. Macro reduced turnover and achieved the
best drawdown. This pattern is consistent with noisy text causing unnecessary
reallocation, transaction costs, and exposure changes during adverse periods.

Turnover is not a complete explanation: risk/uncertainty lowered turnover but
still had a slightly worse drawdown. The timing and direction of trades,
rather than their average volume alone, also matter.

### 5. Macro text is naturally aligned with risk control

Macro stress and uncertainty are broad, slow-moving signals shared across
stocks. They are therefore unlikely to provide strong cross-sectional stock
selection. They are more naturally useful for changing the invested fraction,
cash allocation, rebalance speed, or risk budget.

The macro group's best drawdown and below-baseline turnover support this
interpretation. Feeding macro text into every stock-level observation may
repeat the same information many times; routing it only to the CHRL root
risk/cash decision could use the signal more efficiently.

### 6. Smaller groups reduce estimation and optimization noise

The all-text model adds ten weak and partly overlapping features. With one
seed and finite PPO trajectories, extra dimensions increase the number of
spurious interactions the policy and value networks can fit. Redundant
features can also make credit assignment less stable because PPO must infer
which correlated observation caused a reward change.

Every selected CHRL subset except sentiment had a better return than all
text10. This supports feature dilution or interference as an explanation,
although it does not prove it. The result favors compact groups with explicit
economic roles over unrestricted concatenation.

### 7. CHRL is mechanically better suited to weak text signals

CHRL uses normalized observations, constrained portfolio-weight actions,
turnover/drawdown/action-change penalties, a risk/cash root decision, and
scheduled stock allocation. These mechanisms limit how strongly a noisy input
can change the executed portfolio.

The older flat PPO was documented as more exposed to action clipping and
execution artifacts. Adding weak text dimensions to that policy can change
Gaussian actions without giving the network a stable mapping from textual
meaning to feasible portfolio weights. This can amplify boundary saturation,
unstable trading, and value-function error.

Thus, the old PPO failure does not necessarily mean its text contains no
information. It may mean that the policy and action representation cannot
convert weak, sparse information into controlled portfolio changes.

### 8. Validation and frozen OOS point to possible regime dependence

For the full CHRL text10 model, mean walk-forward validation return and Sharpe
were worse than baseline, while the frozen `fold_2021` OOS result was slightly
better. This disagreement has several possible explanations:

- the 2022-2023 test regime made textual risk and earnings information more
  valuable than earlier validation periods;
- the observed OOS gain is seed variance or sampling noise;
- feature effects are unstable across time;
- model selection indirectly favored behavior that happened to fit the frozen
  period.

Because only one frozen fold and seed are available, regime dependence and
random variation cannot currently be separated. This is the main reason the
positive R6c result should remain a hypothesis.

### 9. Overlapping groups limit feature-level attribution

The groups are not disjoint. `text_signal_confidence` appears in both
sentiment and forward/earnings, while `text_uncertainty` appears in both risk
and macro. Therefore, the ablation identifies useful combinations, not the
unique contribution of each feature.

For example, the forward/earnings result could be driven by numeric evidence,
by the interaction between confidence and earnings pressure, or by all three.
A leave-one-feature-out ablation within that group is required before claiming
that earnings pressure itself is the source of the gain.

## Overall Conclusion

The experiments do **not** establish that adding text generally improves PPO.
They show a narrower result:

1. Raw or broadly concatenated text is harmful to the older flat PPO.
2. Text can provide a modest relative improvement in CHRL.
3. Forward-looking earnings information with numeric evidence and confidence
   is the most promising feature group.
4. Macro text is better suited to drawdown control or regime gating.
5. Generic sentiment features are noisy and offer the weakest risk-adjusted
   trade-off.
6. More text is not better: selected groups outperform the full text10 set.

The practical conclusion is to retain only compact, action-aligned text
features, prioritizing forward/earnings signals and testing macro/risk signals
as gates. Generic sentiment and unrestricted raw text concatenation should not
be used in the final model without stronger evidence.

## Limitations and Decision

The CHRL findings come from one frozen OOS fold and apparently one training
seed. All cumulative returns remain negative, feature groups overlap, and no
confidence intervals or significance tests are available. The observed gains
are therefore screening evidence rather than proof of robust alpha.

The forward/earnings group should advance to multi-seed, multi-fold
confirmation. The current full text10 feature set should not be declared
successful, and the older flat PPO text configuration should be rejected.

## Diagnostic Tests for the Proposed Explanations

| Hypothesis | Test | Confirming result |
| --- | --- | --- |
| Forward signal depends on evidence/confidence | Leave one feature out of the three-feature forward group | Removing evidence density or confidence materially reduces OOS Sharpe |
| Text value is conditional rather than linear | Add explicit earnings x confidence and earnings x evidence interactions to a simple model | Interactions are stable while marginal coefficients remain weak |
| Boilerplate is a reliability gate | Compare raw text with text multiplied by `(1 - boilerplate)` | Gated features improve stability across seeds/folds |
| Sentiment causes excess trading | Compare daily action changes and transaction costs with baseline | Sentiment effects concentrate on high-turnover, negative-advantage days |
| Macro belongs at the risk root | Route macro features only to invested fraction/cash control | Drawdown improves without degrading stock-selection return |
| Full text10 suffers from feature interference | Compare equal-sized random subsets and economically grouped subsets | Economic groups consistently outperform random or full sets |
| CHRL benefit is architectural | Run the same normalized compact features through flat and CHRL policies with matched seeds | CHRL retains the gain while flat PPO remains unstable |
| Frozen OOS gain is regime-specific | Report results by volatility, drawdown, earnings season, and calendar year | Improvement is concentrated in identifiable regimes and repeats out of sample |
