# Methodology: Market-Mechanism-Grounded Interpretable PPO Trading via Self-Supervised Strategy Discovery

Status: canonical methodology spec for the renewed Joseph collaboration.

Date fixed: 2026-05-14.

Relationship to the previous plan:

- This file extends `PROPOSAL_ALIGNED_INTERPRETABILITY_RESEARCH_PLAN.md`.
- If the two files conflict, this file is the more specific implementation and methodology contract.
- Earlier G0/G1/G2 latent-action experiments are reference material only.
- The project is currently at Stage 0, so the immediate priority is PPO baseline quality, instrumentation, feature integrity, and export contracts.

Do not implement the full repository structure while the current model training is running. The structure below is a target architecture for staged migration.

---

# 0. Core Research Claim

## Weak Claim, Guaranteed Target

We discover recurring behavior primitives in a trained PPO trading agent and label them using finance-grounded and market-mechanism-grounded evidence.

## Medium Claim, Target After Diagnostics

We show that primitives are predictive of policy outputs, action changes, turnover, sector exposures, risk, and PPO optimization signatures.

## Strong Claim, Only If Gates Pass

We identify which primitives are causally controllable via safe, in-manifold, sequential interventions.

## Do Not Claim

Do not claim that every primitive is causal. Some primitives may be descriptive clusters, market-regime artifacts, execution artifacts, or previous-position persistence.

The central methodological ladder is:

```text
descriptive structure
→ predictive structure
→ PPO-mechanistic structure
→ causal / controllable structure
```

Main project idea:

> We do not immediately claim that PPO became interpretable. We build a self-supervised vocabulary of behavior primitives, connect primitives to market mechanisms, PPO objective diagnostics, portfolio behavior, and sequential response. Then we test which primitives are merely descriptive, which are predictive, and which can become safe causal levers.

---

# 1. Project Framing

The proposal states that existing interpretability tools such as SHAP, attention, and saliency identify features, but not the strategy the agent executes. Therefore, the project factorizes hidden states into a discrete vocabulary of behaviors and labels those behaviors using finance and ML interpretability methods.

The renewed framing is:

```text
Observation x_t
    ↓
PPO representation h_t
    ↓
self-supervised primitive c_t
    ↓
PPO output: action mean / entropy / value / logprob
    ↓
portfolio behavior: Δweights / turnover / sector exposure / cash
    ↓
market-mechanism label
    ↓
short-horizon and long-horizon outcome
    ↓
causal audit / sequential response audit
```

The project is no longer only:

```text
find labels for hidden-state clusters
```

It becomes:

> Identify market-mechanism primitives inside a PPO trading policy and test whether they are descriptive, predictive, or causally controllable.

This fits the self-supervised requirement because the core representation discovery uses unlabeled trajectories, hidden states, and behavior windows rather than human strategy labels. The course motivation for self-supervised projects is applied work using SSL on a new domain. The latent-variable framing is also relevant because latent variables can add modularity or interpretability, while remaining non-unique.

---

# 2. Literature Grounding

Use these sources as conceptual anchors.

| Component | Literature basis | How it maps to project |
|---|---|---|
| PPO mechanism | PPO alternates between sampling trajectories and optimizing a surrogate objective; clipped PPO allows multiple minibatch epochs while controlling policy update size. ([arXiv][1]) | Log advantage, value error, entropy, logprob, approximate KL, clip fraction by primitive. |
| VQ-VAE primitives | VQ-VAE learns discrete latent representations and uses vector quantization to avoid some posterior-collapse issues. ([arXiv][2]) | Discretize PPO hidden states / behavior windows into primitives. |
| BeXRL baseline | BeXRL discovers behavioral segments for explainable RL, arguing that long trajectory-level explanations can mix multiple behaviors. ([arXiv][3]) | Extend behavior discovery to continuous-action portfolio PPO and automatic finance labels. |
| TCAV / probes controls | Probe control tasks test whether a probe reflects the representation or simply memorizes the task. ([arXiv][4]) | Add probe selectivity, random concepts, temporal holdout. |
| Causal abstraction / intervention | Interchange interventions test whether neural representations have the causal properties of aligned variables. ([arXiv][5]) | Hidden-state interventions are used only as a gated causal audit, not proof by default. |
| Concept bottleneck / adapter | Concept Bottleneck Models allow interventions on predicted concepts and propagate those edits to predictions. ([arXiv][6]) | Future primitive-aware adapter: frozen PPO body plus small trainable correction layer. |
| Market impact decay | Transient impact models represent price impact as past order signs weighted by a propagator / decay function. ([arXiv][7]) | Justifies decay-gated Stage 5.5 interventions. |
| Hawkes re-triggering | Hawkes processes are used in finance for transaction-level volatility, stability, systemic risk, execution, and order-book dynamics. ([arXiv][8]) | Justifies re-triggered intervention strength when bad primitive reappears. |
| Agent-based market simulation | ABIDES supports many trading agents, exchange agent, latencies, and ITCH/OUCH-style messages. ([arXiv][9]) | Future validation in interactive market simulator. |
| Time-series SSL | TS2Vec learns hierarchical contrastive time-series representations at arbitrary semantic levels. ([arXiv][10]) | Optional market-state SSL encoder. |
| JEPA / latent world model | I-JEPA predicts target-block representations from context rather than reconstructing pixels. ([arXiv][11]) | Use latent prediction of future market states instead of raw price reconstruction. |
| Financial LLMs | FinBERT improves financial sentiment analysis with domain-specific language modeling; FinGPT emphasizes open financial LLM data pipelines. ([arXiv][12]) | LLMs should encode news/event context, not directly trade. |
| Financial validation risk | Purged CV reduces leakage when labels overlap through time. Deflated Sharpe adjusts for selection bias, overfitting, sample length, and non-normality. ([Wikipedia][13], [Wikipedia][14]) | Use finance-safe validation for all P&L claims. |
| PPO implementation / action bounding | SB3 recommends observation normalization for PPO/A2C, warns that Gaussian actions are unbounded and clipping is only a bandage, and exposes `target_kl` because PPO clipping alone may not prevent large policy updates. ([SB3 tips][15], [SB3 PPO][16]) | Motivates Stage 0.1: normalized state, softer PPO updates, per-sample PPO diagnostics, and bounded/simplex-aware action semantics. |
| Weight-based portfolio RL | FinRL's portfolio-allocation formulation uses portfolio weights in `[0, 1]` normalized to sum to one, rather than share-trade impulses. ([FinRL][17]) | Supports moving the teacher from trade-signal actions to direct target weights plus cash. |
| Dirichlet portfolio policy | Yang, Park, and Lee model the portfolio vector with a Dirichlet distribution over the simplex and sample low/mid/high-risk portfolios from the learned distribution. ([Yang et al. 2022][18]) | Supports a simplex-aware stochastic policy and risk-level portfolio selection. |
| Dirichlet-tree distribution | The Dirichlet-tree distribution provides a tree-structured composition model with probability splits along a hierarchy. ([Minka 1999][19]) | Mathematical basis for a hierarchical sector-to-stock allocation policy: first allocate to cash/sectors, then allocate within sectors. |
| Public investor holdings / risk preference learning | SEC Form 13F gives delayed quarterly long-position snapshots for institutional managers; inverse optimization can learn risk preferences from observed portfolios. ([SEC Investor.gov][20], [arXiv][21]) | Future extension: investor-style prototypes for labeling or strategy priors, not a first-pass daily control signal. |

---

# 3. Definitions

Codex should standardize these names.

```python
x_t          # PPO observation at date t
h_t          # PPO hidden representation / policy_latent
v_t          # value latent, if separable
a_t          # original PPO action
mu_t         # policy action mean
sigma_t      # policy action std
target_w_t   # target portfolio weights proposed by a weight-based policy
w_exec_t     # executed portfolio weights after smoothing, constraints, and costs
g_t          # scalar or vector rebalance gate
e_t          # portfolio tracking error target_w_t - w_{t-1}
sector_w_t   # top-level cash/sector allocation in a hierarchical policy
stock_w_t    # within-sector stock allocation in a hierarchical policy
alpha_t      # positive concentration parameters for Dirichlet/simplex policy
logprob_t    # log probability of executed action
entropy_t    # policy entropy
value_t      # PPO value estimate
adv_t        # advantage estimate
ret_t        # realized reward / portfolio return
w_t          # portfolio weights
dw_t         # weight delta
turnover_t   # L1 weight change
c_t          # discovered primitive id
z_market_t   # optional SSL market-state embedding
m_t          # market mechanism label
r_t          # market regime label
ood_t        # hidden-state OOD distance
```

Primitive:

```text
A primitive is a recurring discrete behavior code learned from PPO representations and/or behavior windows.
```

Market mechanism:

```text
A market mechanism is a hypothesized real-market pattern that may explain why a primitive appears:
momentum, reversal, risk-off, sector rotation, liquidity stress, execution unwind, news reaction, crowding, etc.
```

Intervention:

```text
An intervention modifies hidden state, input features, subspace, action logits, or an adapter output to test whether a primitive is controllable.
```

---

# 4. Required Data Logging

Before any SSL or causal audit, Codex must create a complete trajectory log.

## 4.1 Required Per-Timestep Log

For every trading day `t`, save:

```python
{
    "date": date,
    "split": "train/val/test/oos",
    "obs": x_t,
    "policy_latent": h_t,
    "value_latent": v_t_or_none,
    "hidden_layers": optional_list,
    "action": a_t,
    "action_mean": mu_t,
    "action_std": sigma_t,
    "logprob": logprob_t,
    "entropy": entropy_t,
    "value": value_t,
    "reward": reward_t,
    "return_1d": ret_t,
    "return_5d": optional,
    "return_10d": optional,
    "return_20d": optional,
    "advantage": adv_t,
    "approx_kl": approx_kl_t,
    "clip_fraction": clip_indicator_t,
    "portfolio_weights": w_t,
    "weight_delta": dw_t,
    "cash": cash_t,
    "gross_exposure": gross_t,
    "net_exposure": net_t,
    "turnover": turnover_t,
    "transaction_cost": tc_t,
    "drawdown": dd_t,
    "sector_exposures": sector_vector_t,
    "factor_features": factor_vector_t,
    "macro_features": macro_vector_t,
    "market_regime": optional_r_t,
    "news_event_embedding": optional_news_t
}
```

## 4.2 Important Implementation Rule

Use exact PPO observations, not only engineered proxy-state features. Previous latent-actions work showed that exact PPO observations often contain more useful signal than aggregated proxies. Proxy features are allowed for labels and diagnostics, but not as a replacement for PPO inputs.

---

# 5. Stage 0 — PPO Baseline and Instrumentation

## Goal

Train or load the frozen PPO trading agent and generate trajectory logs.

## Required Outputs

```text
artifacts/stage0/ppo_model.zip
artifacts/stage0/trajectory.parquet
artifacts/stage0/portfolio_timeseries.parquet
artifacts/stage0/ppo_diagnostics.parquet
artifacts/stage0/config.yaml
```

## PPO Diagnostics to Compute

For each timestep and later by primitive:

```text
advantage
value estimate
value error
entropy
logprob
approx KL
clip fraction
reward-to-go
action mean
action std
turnover
transaction cost
drawdown
```

Reason: PPO interpretability must connect primitives to PPO's own optimization terms, not only to portfolio outcomes. PPO's original paper defines PPO as a policy-gradient method based on surrogate objectives and repeated minibatch updates, so primitive diagnostics should include quantities that reveal whether a primitive coincides with negative advantage, low entropy, clipping, or value error. ([arXiv][1])

---

# 5A. Stage 0.1 — Stabilized Interpretable PPO Teacher

## Motivation

The current Stage 0 teacher is useful as a first frozen policy for exploratory primitive discovery, but it is mechanically unstable: the Gaussian PPO policy often produces raw actions outside the action box, the environment clips them, and the executed behavior can become dominated by boundary saturation, cash constraints, holdings, and execution mechanics.

For interpretability this is not a minor engineering issue. If a teacher learns to push actions to the boundary and lets the environment resolve the constraint, Stage 1 primitives may describe clipping/execution artifacts rather than investment behavior. Stage 0.1 therefore creates a second, more stable teacher while Stage 1 discovery may continue on the current teacher.

## Core Design Rule

Do not treat action clipping as the portfolio policy. The stabilized teacher should speak the native language of portfolio management:

```text
policy output = target portfolio weights over 29 stocks + cash
```

The action space is a simplex:

```text
w_t in R_+^{N+1}
sum_i w_{i,t} = 1
```

where the extra coordinate is cash. For the current 29-stock universe:

```text
N = 29 stocks
N + 1 = 30 portfolio coordinates including cash
```

This turns "risk-off" into an explicit allocation to cash rather than an indirect saturating sell impulse.

## State Redesign Rule

The stabilized policy should not receive raw accounting magnitudes as unnormalized decision features. Raw cash, raw prices, and raw shares are poor neural inputs because their absolute units are not the market mechanism. Use features that describe market state, portfolio state, and changes:

```text
price inputs:
  log_return_1d, log_return_5d, log_return_20d
  momentum
  drawdown from rolling high
  price_to_moving_average_minus_1

volume/liquidity inputs:
  volume_zscore
  volume_change
  dollar_volume
  high_low_range
  Amihud-style illiquidity proxy if available

volatility/regime inputs:
  realized_vol
  VIX level, VIX change, VIX percentile
  HMM/regime probabilities, regime deltas, regime entropy

portfolio-state inputs:
  previous stock weights
  cash weight
  turnover
  concentration / HHI
  current drawdown
```

Important: not every feature should become a relative change. Some levels are semantically meaningful: VIX level, RSI level, cash weight, previous weights, drawdown level, and regime probabilities are state variables, not just changes. The default pattern is:

```text
level when the level has economic meaning
change when the transition has economic meaning
train-only z-score / percentile for scale control
```

Use train-only normalization statistics and freeze them for validation/test.

## Simplex-Aware Policy Options

Stage 0.1 should compare these action parameterizations in increasing complexity:

```text
A. Softmax-projected simplex policy
   network logits -> softmax -> target weights
   Fastest fallback, easy to debug, deterministic-friendly.

B. Flat Dirichlet policy
   alpha_t = softplus(policy_head(h_t)) + alpha_min
   target_w_t ~ Dirichlet(alpha_t)
   E[target_w_t] = alpha_t / sum(alpha_t)
   Proper stochastic distribution over portfolio weights.

C. Hierarchical sector-stock Dirichlet policy
   sector/cash level first, within-sector stock level second.
   Main interpretability candidate.
```

Avoid returning to independent Gaussian coordinates plus clipping as the main teacher unless this is explicitly marked as a negative control.

## Hierarchical Dirichlet Policy

Let:

```text
G       = number of top-level groups
g       = group index
N_g     = number of tradable stocks inside group g
cash    = a top-level group with one asset
```

Top-level groups should include cash and economically meaningful sectors:

```text
cash
energy
financials
technology
healthcare
industrials
consumer
...
```

The policy network maps the current representation `h_t` or observation encoder output into positive concentration parameters:

```text
alpha_sector_t = softplus(f_sector(h_t)) + alpha_min
alpha_stock_{g,t} = softplus(f_stock_g(h_t)) + alpha_min
```

Top-level allocation:

```text
s_t ~ Dirichlet(alpha_sector_t)
sum_g s_{g,t} = 1
```

Within-sector allocation:

```text
u_{g,t} ~ Dirichlet(alpha_stock_{g,t})
sum_{i in g} u_{i|g,t} = 1
```

Final stock weights:

```text
w_{i,t} = s_{g(i),t} * u_{i|g(i),t}
```

Cash:

```text
w_cash,t = s_cash,t
```

Final constraint:

```text
sum_i w_{i,t} + w_cash,t = 1
```

This is one portfolio decision, not a mixture of alternative strategies. If the model allocates simultaneously to financials and energy, that means both sector mechanisms are active in the same portfolio:

```text
cash        15%
financials 30%
energy     25%
technology 10%
healthcare 10%
other       10%
```

This should be interpreted as a mixed sector allocation, not as the agent following several mutually exclusive strategies.

## Dirichlet-Tree Interpretation

The hierarchical policy can be viewed as a practical portfolio version of a Dirichlet-tree composition:

```text
root
  -> cash
  -> sectors
       -> stocks inside each sector
```

Each internal node splits its parent budget with a Dirichlet distribution. The final leaf weights are products of the splits along the path from root to leaf:

```text
w_leaf = product_of_branch_probabilities(root -> ... -> leaf)
```

This gives a clear mathematical foundation for "first choose cash/sector allocation, then choose stocks within each sector." It also gives two levels of interpretability:

```text
sector/cash mechanism:
  risk-off, energy-heavy, financials-heavy, broad defensive, etc.

within-sector stock selection:
  broad sector allocation vs concentrated stock pick
```

## PID-Inspired Execution Layer

The policy should propose target weights. A separate execution layer should convert target weights into executed weights smoothly:

```text
e_t = target_w_t - w_{t-1}
```

The simplest version is the inertial rebalance gate:

```text
w_exec_t = (1 - g_t) * w_{t-1} + g_t * target_w_t
```

Equivalent P-controller form:

```text
delta_w_t = Kp * e_t
w_exec_t = ProjectToSimplex(w_{t-1} + delta_w_t)
```

Compare four controller families:

```text
P:
  delta_w_t = Kp * e_t

PI:
  delta_w_t = Kp * e_t + Ki * sum_{j<=t} e_j

PD:
  delta_w_t = Kp * e_t + Kd * (e_t - e_{t-1})

PID:
  delta_w_t = Kp * e_t + Ki * sum_{j<=t} e_j + Kd * (e_t - e_{t-1})
```

Implementation constraints:

```text
ProjectToSimplex after the controller.
Add turnover cap.
Add anti-windup for PI/PID.
Clip or smooth derivative term for PD/PID.
Log all controller terms per timestep.
```

Expected trade-offs:

```text
P:
  simplest and most stable baseline.

PI:
  helps if the agent persistently fails to reach target weights.
  risk: integral windup and excessive delayed trading.

PD:
  main smooth-trading candidate.
  helps reduce overshoot, churn, and sudden turnover.
  risk: derivative term can amplify noisy targets.

PID:
  most flexible, but easiest to overcomplicate.
  run only after P/PI/PD are stable.
```

The initial experimental priority is:

```text
1. Flat Dirichlet + P
2. Hierarchical Dirichlet + P
3. Flat Dirichlet + PD
4. Hierarchical Dirichlet + PD
```

Expand to PI/PID only after the P/PD results are mechanically healthy.

## Reward and Realism

The previous custom/Zhang rewards partially addressed return scaling and volatility, and transaction costs entered through the environment's asset value. They did not fully solve the execution artifact problem because they did not explicitly penalize all relevant portfolio mechanics.

Stage 0.1 reward should include:

```text
reward_t =
    risk_adjusted_return_t
  - lambda_turnover      * turnover_t
  - lambda_drawdown      * drawdown_increment_t
  - lambda_concentration * concentration_t
  - lambda_action_change * ||w_exec_t - w_{t-1}||_1
```

The exact coefficients must be selected by walk-forward validation, not by frozen test.

## PPO Stabilization Settings

The stabilized PPO configuration should start conservatively:

```text
target_kl: around 0.01 as first value
n_epochs: 3 to 5 before trying larger values
learning_rate: around 1e-4 or lower as first value
clip_range: around 0.10 to 0.15
ent_coef: very small, with decay toward zero
max_grad_norm: keep enabled
actor/critic: separate networks, compact actor, critic may be wider
value_clipping: not default
```

For any Gaussian fallback:

```text
negative log_std_init
monitor std trend
fail run if raw action saturation persists
```

## Per-Timestep Instrumentation for Stage 0.1

Batch-mean PPO logs are insufficient. Save per-timestep or per-sample values:

```text
date
split
observation_id
raw_policy_params
target_weights
executed_weights
previous_weights
portfolio_tracking_error
controller_p_term
controller_i_term
controller_d_term
simplex_projection_residual
turnover
transaction_cost
cash_weight
sector_weights
within_sector_weights
old_logprob
new_logprob
probability_ratio
clip_indicator
approx_kl_sample
entropy
value_old
value_new
GAE_advantage
reward_to_go
value_error
policy_loss_sample
value_loss_sample
execution_constraint_flags
```

This data is required so that later primitives can be diagnosed as:

```text
portfolio strategy primitive
sector rotation primitive
risk-off/cash primitive
execution artifact
controller artifact
policy-instability artifact
```

## Stage 0.1 Acceptance Criteria

The stabilized teacher is acceptable only if it passes both performance and mechanical checks:

```text
Performance:
  validation-selected only
  frozen test used once
  does not lose to equal-weight and DJI on key risk-adjusted metrics

KL:
  mean approx_kl normally < 0.01
  95th percentile normally < 0.02
  repeated spikes > 0.05 are fail signals

Clip regime:
  clip_fraction should not live near 1.0
  sustained > 0.4 is a warning

Simplex health:
  weights sum to 1 after execution
  no near-one-hot collapse without economic reason
  cash allocations are plausible during risk-off periods

Turnover:
  materially lower than saturating share-trade policy
  not dominated by exploration noise

Action fidelity:
  target weights and executed weights should be close except for documented costs,
  caps, market drift, or controller smoothing

Hidden-state readiness:
  primitives should not be dominated by action saturation, projection residual,
  or controller failure flags
```

## Optional Strategy-Level Extension

A higher-level strategy allocation can be added later, but it is not required for the first stabilized teacher.

Possible future structure:

```text
strategy weights rho_t ~ Dirichlet(alpha_strategy_t)

strategy prototypes:
  conservative / value-quality
  moderate / diversified
  aggressive / growth
  macro/risk-off
  sector-specialist

w_strategy_t = sum_k rho_{k,t} * prototype_portfolio_{k,t}
w_target_t = lambda_t * w_strategy_t + (1 - lambda_t) * w_direct_policy_t
```

Investor-style prototypes could be learned from public institutional holdings, mutual-fund holdings, or 13F snapshots using SSL methods such as VQ-VAE, sequence autoencoders, clustering, or contrastive encoders. However, this must be treated as a labeling/prior extension, not as clean daily trading supervision:

```text
13F is quarterly and delayed.
13F mostly captures long public positions.
It omits or weakly captures cash, shorts, derivatives, private holdings, and intra-quarter trades.
Famous-investor names may map to organizations, teams, mandates, or funds rather than one decision maker.
VC investors are often poor analogues for daily public-equity allocation.
```

Therefore, investor-style SSL is useful for:

```text
external style benchmarks
primitive labeling
risk-preference priors
strategy-prototype adapters
```

but it should not replace the Stage 0.1 hierarchical Dirichlet teacher.

---

# 6. Stage 1 — Self-Supervised Primitive Discovery

The current proposal uses VQ-VAE on penultimate hidden states `h(t)` to create primitive sequence `c(t)` and checks utilization, perplexity, reconstruction, temporal stability, and cross-fold NMI. Keep this, but expand it.

## 6.1 Models to Train

Codex should train and compare these variants:

```text
A. Hidden-only VQ-VAE
   input = h_t

B. Windowed transformer VQ-VAE
   input = [h_{t-L+1}, ..., h_t]

C. Behavior-window VQ-VAE
   input = [w_t, Δw_t, turnover_t, cash_t, sector_exposure_t]

D. Joint policy-behavior VQ-VAE
   input = concat(h_t, Δw_t, turnover_t, sector_exposure_t)

E. Joint policy-market VQ-VAE
   input = concat(h_t, z_market_t, Δw_t, turnover_t)
```

Recommended first pass:

```text
A + B + D
```

If time permits:

```text
E with optional SSL market encoder
```

## 6.2 VQ-VAE Details

Use standard VQ-VAE:

```text
encoder(x) → z_e
nearest codebook vector e_k
decoder(e_k) → reconstruction
loss = reconstruction_loss + β * commitment_loss
```

The course notes state that VQ bottlenecks are a dominant approach for discrete latent variables and list improvements such as cosine similarity, residual VQ, stale-code expiration, k-means initialization, orthogonal regularization, and multi-headed VQ. Implement at least:

```text
k-means or k-means++ codebook initialization
EMA codebook updates
stale-code reset
cosine similarity option
K-ablation
```

## 6.3 Hyperparameter Grid

```yaml
K: [16, 32, 64, 128]
window_lengths: [1, 3, 5, 10, 20]
commitment_beta: [0.1, 0.25, 0.5, 1.0]
distance: ["l2", "cosine"]
model_type: ["mlp_vqvae", "transformer_vqvae"]
```

## 6.4 Stage 1 Success Gates

Keep original gates from the proposal:

```text
codebook utilization ≥ 0.7
perplexity ≥ K / 2
reconstruction error reported as fraction of h variance
median primitive run length ≥ 3 trading days
cross-fold NMI ≥ 0.4
K-ablation must not be dropped
```

Add new gates:

```text
action fidelity lift > previous-code baseline
primitive distinctiveness > HMM-regime-only baseline
primitive stability on test split
no single primitive dominates > 50% of timesteps unless explicitly flagged
```

## 6.5 Negative Controls

Codex must run:

```text
random codebook
shuffled primitive labels
HMM-only regime labels
k-means on portfolio weights
previous-code Markov baseline
random hidden directions
```

Purpose: prevent beautiful clusters from being mistaken for interpretable behavior.

---

# 7. Stage 2 — Action Fidelity and Portfolio Diagnostics

The current proposal's Stage 2 computes mean weights, sector concentration, cash, gross exposure, turnover, HHI, and per-primitive returns. Keep all of that, but add policy-output fidelity.

## 7.1 Portfolio Diagnostics

For every primitive `k`:

```text
mean portfolio weights
mean cash
gross exposure
net exposure
turnover
transaction cost
HHI concentration
sector exposure vector
sector turnover
benchmark deviation
return distribution: mean, std, skew, kurtosis
drawdown contribution
```

## 7.2 Action Fidelity Diagnostics

For every primitive `k`:

```text
mean action
mean action delta
cosine similarity of actions within primitive
R²(action_mean ~ primitive_id)
R²(weight_delta ~ primitive_id)
KL(policy(a|h_t) || primitive-conditioned policy proxy)
behavior-cloning MSE from primitive-only model
behavior-cloning MSE from primitive + market-state model
lift over previous-code baseline
```

## 7.3 Required Comparison

```text
primitive model must beat:
1. previous-code baseline
2. HMM-regime baseline
3. market-state-only baseline
4. portfolio-weight k-means baseline
```

If it does not beat these, the primitive is probably a persistence or regime artifact.

---

# 8. Stage 3 — Finance-Grounded and Market-Mechanism-Grounded Labeling

The original Stage 3 uses five labeling methods: RBSA, classical-signal correlation, linear probes/TCAV, HMM regime alignment, and synthetic-strategy correlation. It labels a primitive as well-labeled if at least 3 of 5 methods agree. Keep this as the finance label layer.

Add a second label layer: Market Mechanism Bank.

## 8.1 Original Five Finance Labels

Keep:

```text
1. Returns-Based Style Analysis
2. Classical signal correlations
3. Linear probes and TCAV
4. HMM regime alignment
5. Synthetic strategy correlation
```

Keep convergent-evidence rule:

```text
≥3 methods agree → strong label
2 methods agree → tentative label
≤1 method agrees → novel / uninterpretable / inspect manually
```

## 8.2 Add Market-Mechanism Labels

For each primitive, Codex should score the following mechanisms:

| Mechanism | Observable proxies | Expected primitive signature |
|---|---|---|
| Momentum / trend following | 12-1 momentum, market trend, sector momentum | buys winners, sells losers, bull-regime affinity |
| Short-term reversal | 1d/5d reversal, prior drawdown | buys dips, sells rallies |
| News underreaction | LLM/FinBERT event signal, post-news drift | gradual exposure shift after news |
| Overreaction / reversal risk | extreme return, attention shock, sentiment spike | sharp trade, later reversal or drawdown |
| Risk-off deleveraging | VIX, realized volatility, drawdown, correlation spike | raises cash, lowers gross exposure |
| Volatility targeting | realized vol/VIX up → exposure down | systematic risk scaling |
| Sector rotation | sector momentum, macro factors, sector exposure shift | rotates capital between sectors |
| Crowding / herding | high concentration, high benchmark deviation, high correlation regime | chases popular exposure |
| Liquidity stress | volume shock, spread proxy, high-low range, Amihud illiquidity | high transaction-cost sensitivity |
| Execution / metaorder artifact | repeated same-sign weight deltas, action clipping, cumulative turnover | slow unwind or repeated rebalancing |
| Forced liquidation / stop-loss | drawdown + sell after selloff + high turnover | sells into weakness |
| Calendar / institutional rebalancing | month-end, quarter-end, earnings season | predictable turnover around calendar events |

## 8.3 Mechanism Score

For each primitive `k` and mechanism `m`, compute:

```text
mechanism_score(k, m) =
    evidence_from_signals
  + evidence_from_portfolio_behavior
  + evidence_from_PPO_diagnostics
  + evidence_from_temporal_signature
  + evidence_from_regime_or_sector_conditioning
```

Save:

```text
artifacts/stage3/primitive_labels.csv
artifacts/stage3/mechanism_scores.csv
artifacts/stage3/label_uncertainty.csv
```

## 8.4 LLM / News Layer

Use LLMs only as event encoders, not as trading agents.

For each date / stock / sector:

```json
{
  "event_type": "earnings_guidance | macro | litigation | M&A | downgrade | product | regulation | none",
  "sentiment": "positive | negative | neutral",
  "salience": 0.0,
  "uncertainty": 0.0,
  "novelty": 0.0,
  "affected_tickers": [],
  "affected_sectors": [],
  "expected_response_shape": "shock_decay | drift | reversal | mixed"
}
```

Rationale: FinBERT shows domain-specific language models are useful for financial sentiment, while FinGPT frames financial LLMs as data-centric tools for finance workflows rather than guaranteed trading oracles. ([arXiv][12])

Strict rule:

```text
No look-ahead timestamps.
No article published after market close may affect same-day action unless data pipeline explicitly supports it.
```

---

# 9. Stage 4 — Outcome Analysis and PPO-Mechanism Diagnostics

Original Stage 4 decomposes agent return into per-primitive contributions and flags performance-positive and performance-negative behaviors. Keep that, but do not stop at return attribution.

## 9.1 Return Decomposition

For each primitive:

```text
sum of one-period returns
mean return
volatility
Sharpe
Sortino
max drawdown
transaction-cost-adjusted return
factor-adjusted alpha
contribution to total volatility
contribution to total drawdown
```

## 9.2 PPO Diagnostics by Primitive

For every primitive `k`:

```text
mean advantage
advantage distribution
value error
entropy
logprob
clip fraction
approx KL
reward-to-go
action std / confidence
turnover cost
drawdown contribution
```

Interpretation examples:

```text
negative advantage + high turnover + low entropy
→ likely bad overconfident primitive

high value error + high drawdown
→ value-network failure primitive

high clip fraction + repeated same-sign Δweights
→ execution/action-clipping artifact

low return but lower drawdown
→ possibly protective risk-off primitive, not bad
```

## 9.3 Finance-Safe Statistical Validation

Codex should implement:

```text
walk-forward split
purged CV with embargo for overlapping return horizons
block bootstrap for time-series uncertainty
HAC/Newey-West errors for regressions
Deflated Sharpe Ratio for multiple strategy / primitive searches
Bonferroni or FDR for signal correlations
```

Reason: financial ML is vulnerable to leakage, overfitting, overlapping labels, non-normal returns, and multiple testing. Purged cross-validation is designed for financial time series where labels depend on future events and overlap across folds. Deflated Sharpe adjusts for selection bias, backtest overfitting, sample length, and non-normality. ([Wikipedia][13], [Wikipedia][14])

---

# 10. Stage 4.5 — Mechanism-Conditioned Diagnostics

This is a bridge stage.

Goal:

> Decide whether a primitive is a real market-mechanism candidate before causal interventions.

For each primitive × market mechanism:

```text
primitive_id
mechanism_label
mechanism_score
advantage_mean
value_error
entropy
turnover_cost
sector_exposure_shift
risk_state
news_event_context
return_3d
return_5d
return_10d
drawdown_10d
ood_risk
status
```

Status categories:

```text
GOOD_MECHANISM
BAD_MECHANISM
PROTECTIVE_RISK_MECHANISM
EXECUTION_ARTIFACT
REGIME_ARTIFACT
PERSISTENCE_ARTIFACT
UNINTERPRETABLE
```

Important:

```text
Do not suppress a primitive just because it has negative raw return.
It may be a protective risk-off primitive active during crashes.
```

---

# 11. Stage 5 — One-Step Causal Audit

Before sequential interventions, run one-step causal tests.

## 11.1 Intervention Types

Codex should implement four intervention arms:

```text
A. Hidden direction intervention
B. Hidden subspace intervention
C. Input-level intervention
D. Logit/action-mean intervention
```

Optional future arm:

```text
E. Trainable adapter intervention
```

## 11.2 Hidden Direction Intervention

For a primitive direction `d_bad`:

```text
h'_t = h_t - η * d_bad
```

For steering bad → good:

```text
h'_t = h_t - η_bad * d_bad + η_good * d_good
```

## 11.3 Subspace Intervention

Instead of one direction:

```text
h'_t = h_t + η * P_S(d)
```

where `P_S` projects onto the learned primitive subspace.

Use this when single-vector interventions are unstable or entangled.

## 11.4 OOD Gates

For every modified hidden state:

```text
Mahalanobis distance
kNN distance
local PCA reconstruction error
autoencoder reconstruction error
density percentile
logprob under original policy
action KL from original policy
```

Pass only if:

```text
OOD distance below threshold
action shift not extreme
entropy does not collapse
policy logprob not implausible
```

## 11.5 One-Step Gates

A primitive is eligible for Stage 5.5 only if it passes:

```text
Gate 1: Directionality
  intervention changes primitive probability in intended direction

Gate 2: Policy output
  action mean / entropy / logprob shift is predictable

Gate 3: Portfolio behavior
  turnover, exposure, concentration shift is intended

Gate 4: OOD safety
  h'_t stays near natural hidden manifold

Gate 5: No immediate risk blow-up
  no extreme turnover, exposure, or drawdown proxy
```

If gates fail:

```text
status = descriptive_only_or_unsafe
do not run Stage 6 rollout
```

---

# 12. Stage 5.5 — Market-Mechanism Sequential Response Audit

This is the central new stage.

Goal:

> Test whether primitive interventions produce stable multi-day policy response without leaving the hidden-state manifold.

This stage is not a full rollout claim. It is an impulse-response audit.

## 12.1 Window Selection

Select intervention windows around:

```text
bad primitive activation
p_bad(t) > threshold
high mechanism score
bad primitive dwell episode
large negative advantage primitive episode
high-turnover primitive episode
news/event-conditioned primitive episode
```

Window lengths:

```text
L ∈ {3, 5, 10, 20}
```

## 12.2 Basic Sequential Intervention

For date `τ` and step `s = 0, ..., L-1`:

```text
h'_{τ+s} = h_{τ+s} + η0 * g(s; θ) * m(τ+s) * P(d)
```

For bad primitive suppression:

```text
h'_{τ+s} = h_{τ+s} - η0 * g(s; θ) * m(τ+s) * P(d_bad)
```

For bad-to-good steering:

```text
h'_{τ+s} =
    h_{τ+s}
  - η_bad  * g_bad(s)  * m_bad(τ+s)  * P(d_bad)
  + η_good * g_good(s) * m_good(τ+s) * P(d_good)
```

Where:

```text
g(s; θ) = temporal kernel
m(t)    = market-condition gate
P(d)    = direction or subspace projection
η0      = intervention strength
```

## 12.3 Decay Kernels

Codex should implement:

| Kernel | Formula | Purpose |
|---|---:|---|
| Constant baseline | `g(s)=1` | old naive baseline |
| Linear decay | `g(s)=1-s/(L-1)` | simple decay check |
| Exponential decay | `g(s)=exp(-s/τ)` | shock-response / transient impact |
| Power-law decay | `g(s)=(1+s/c)^(-β)` | slow digestion / long memory |
| Logistic decay | `g(s)=1/(1+exp((s-m)/T))` | plateau then decay |
| Raised cosine | `g(s)=0.5*(1+cos(pi*s/(L-1)))` | smooth shutdown |
| HAR mixture | `Σ_i w_i exp(-s/τ_i)` | daily / weekly / monthly horizons |
| Hawkes re-triggered | `η_t = ρ*η_{t-1} + η0*trigger_t` | bad primitive reappears → intervention reactivates |
| Reverse decay placebo | `g_reverse(s)=g(L-1-s)` | tests whether early force matters |
| Random same-dose | random schedule with same total dose | placebo |

Market-impact literature motivates decay kernels because transient impact models use past order signs weighted by propagator functions. Hawkes processes motivate re-triggered schedules because event intensity can rise after prior events and Hawkes models are used in financial order-flow and execution contexts. ([arXiv][7], [arXiv][8])

## 12.4 Market-Condition Gates

Define:

```text
m(t) = volatility_gate(t)
     * liquidity_gate(t)
     * sector_gate(t)
     * news_gate(t)
     * correlation_regime_gate(t)
     * drawdown_gate(t)
     * OOD_gate(t)
```

Example:

```text
η_t = η0
    * exp(-s / τ)
    * sigmoid(-OOD_distance_t)
    * sigmoid(-volatility_spike_t)
    * sigmoid(liquidity_proxy_t)
```

Meaning:

```text
strong early intervention
decays over time
weaker when OOD risk is high
weaker when liquidity/volatility makes action unsafe
```

## 12.5 Normalize Intervention Budget

Every kernel comparison must be normalized in three ways:

```text
1. Peak-normalized
   same η0

2. Dose-normalized
   same Σ_s |η_s|

3. OOD-normalized
   same max allowed OOD distance
```

Otherwise exponential decay may look safer simply because it applies less total force.

## 12.6 Stage 5.5 Response Metrics

For every window:

```text
primitive probability path:
  p_bad(t+s), p_good(t+s)

primitive dwell:
  dwell time in bad primitive
  AUC_bad = Σ_s p_bad(t+s)

transition behavior:
  bad → neutral
  bad → good
  bad → another bad

policy output path:
  action_mean shift
  action_std shift
  entropy shift
  value shift
  logprob under original policy
  approximate KL

portfolio path:
  turnover
  transaction costs
  sector exposure
  gross exposure
  cash
  HHI
  benchmark deviation

risk path:
  drawdown
  realized volatility
  concentration

OOD path:
  Mahalanobis
  kNN distance
  local PCA error
  autoencoder reconstruction error

outcome path:
  cumulative return 3d / 5d / 10d / 20d
  factor-adjusted abnormal return
  transaction-cost-adjusted return
```

## 12.7 Main Score

```text
Safe Primitive Suppression Score =
    ΔAUC_bad
  - λ1 * ΔOOD
  - λ2 * Δturnover
  - λ3 * Δdrawdown
  - λ4 * |Δsector_exposure_unintended|
  - λ5 * |Δpolicy_KL|
```

## 12.8 Required Controls

Codex must run:

```text
no intervention
constant intervention
decay intervention
reverse decay
random same-dose schedule
wrong-direction intervention
random primitive direction
shifted event window
matched non-bad window
same primitive in different regime
same primitive in different sector regime
OOD-capped vs uncapped
```

Most important test:

```text
same total dose, different temporal shape
```

If early-strong decay beats reverse decay under the same dose and OOD cap, temporal shape matters.

---

# 13. Stage 6 — Primitive-Aware Adapter, Only If Stage 5.5 Succeeds

Do not jump directly to full rollout. Stage 6 only runs if the primitive passes:

```text
Stage 1 coherence
Stage 2 action fidelity
Stage 3 label confidence
Stage 4 PPO diagnostic relevance
Stage 5 one-step causal gates
Stage 5.5 sequential safety gates
```

## 13.1 Adapter Architecture

```text
frozen PPO body
    ↓
policy_latent h_t
    ↓
small trainable adapter A(h_t, c_t, z_market_t)
    ↓
corrected latent h̃_t
    ↓
frozen or lightly trainable PPO policy head
    ↓
action distribution
```

## 13.2 Adapter Loss

```text
L =
    L_behavior_clone_to_original_PPO
  + λ_KL * KL(π_adapter || π_original)
  + λ_bad * bad_primitive_activation_penalty
  + λ_turnover * turnover_penalty
  + λ_concentration * concentration_penalty
  + λ_OOD * hidden_manifold_penalty
  - λ_adv * advantage_consistency_term
```

Concept Bottleneck Models justify this direction because they allow edits to high-level concepts and propagate those edits to predictions. Here the concepts are discovered primitives, but because they are not supervised human concepts, adapter results must be compared against random-label and no-primitive baselines. ([arXiv][6])

## 13.3 Adapter Baselines

```text
original PPO
PPO with global turnover penalty
adapter without primitive penalty
adapter with random primitive labels
adapter with HMM regime labels only
logit-level turnover controller
sector/regime rule baseline
```

## 13.4 Stage 6 Success Criteria

```text
lower bad primitive AUC
lower turnover or drawdown
no large OOD increase
KL to original policy within cap
no deterioration on validation / OOS
improvement survives purged CV / walk-forward test
```

---

# 14. Optional Stage 7 — SSL Market-State Encoder and Latent World Model

This is optional, not required for first implementation.

## 14.1 Market-State SSL Encoder

Train self-supervised encoder:

```text
z_market_t = E_market(
    prices,
    returns,
    volume,
    volatility,
    macro,
    sector features,
    optional news embeddings
)
```

Possible objectives:

```text
masked reconstruction
contrastive predictive learning
TS2Vec-style hierarchical contrastive objective
JEPA-style future latent prediction
```

TS2Vec is relevant because it learns timestamp-level and subsequence time-series representations using hierarchical contrastive learning. JEPA is relevant because it predicts latent representations rather than reconstructing raw observations; I-JEPA uses context blocks to predict target-block representations and emphasizes semantic representations. ([arXiv][10], [arXiv][11])

## 14.2 Latent World Model

Train:

```text
z_market_{t+1:t+H} = WorldModel(
    z_market_t,
    action_t,
    event_t,
    sector_state_t
)
```

Purpose:

```text
simulate counterfactual primitive windows
test interventions under alternative volatility / liquidity / news states
avoid relying only on one historical path
```

Keep this as a future extension unless Stage 5.5 has stable signals.

---

# 15. Market Mechanism Bank

These are the concrete market mechanisms to track.

## 15.1 Momentum / Trend Following

Signals:

```text
12-1 momentum
sector momentum
market trend
return streak
relative strength
```

Primitive signature:

```text
buys recent winners
sells recent losers
low cash in bull regimes
possibly high concentration
```

Risk:

```text
can become crowded momentum or late-cycle overreaction
```

## 15.2 Short-Term Reversal / Liquidity Provision

Signals:

```text
1-day reversal
5-day reversal
prior drawdown
oversold indicators
```

Primitive signature:

```text
buys dips
sells rallies
moderate turnover
possibly stabilizing
```

Risk:

```text
may catch falling knives during crash regimes
```

## 15.3 News Underreaction

Signals:

```text
positive/negative news
earnings guidance
analyst changes
post-news drift
LLM/FinBERT salience
```

Primitive signature:

```text
gradual position adjustment after news
multi-day drift behavior
```

Risk:

```text
leakage from incorrect timestamps
LLM hallucination
news already priced in
```

## 15.4 Overreaction / Reversal Risk

Signals:

```text
large gap
extreme sentiment
attention spike
abnormally high volume
```

Primitive signature:

```text
large immediate action shift
later reversal
high drawdown risk
```

Risk:

```text
mistakenly suppresses profitable momentum
```

## 15.5 Risk-Off Deleveraging

Signals:

```text
VIX
realized volatility
correlation spike
market drawdown
credit spread
```

Primitive signature:

```text
raises cash
lowers gross exposure
rotates into defensive sectors
```

Risk:

```text
looks performance-negative because it activates during bad markets, but may be protective
```

## 15.6 Volatility Targeting

Signals:

```text
realized volatility
VIX
vol-of-vol
drawdown
```

Primitive signature:

```text
systematic exposure reduction when volatility rises
```

Risk:

```text
may over-deleverage near market bottoms
```

## 15.7 Sector Rotation

Signals:

```text
sector momentum
sector macro sensitivity
rates
yield curve
commodity sensitivity
```

Primitive signature:

```text
capital moves between tech, financials, healthcare, industrials, defensives
```

Risk:

```text
DJ30 may be too small for clean sector effects
```

## 15.8 Crowding / Herding

Signals:

```text
sector concentration
benchmark deviation
market breadth
correlation regime
momentum crowding
```

Primitive signature:

```text
chases popular exposures
low entropy
high confidence
```

Risk:

```text
works in bull markets, fails in reversals
```

## 15.9 Liquidity Stress

Signals:

```text
volume shock
spread proxy
high-low range
Amihud illiquidity
realized volatility
```

Primitive signature:

```text
high transaction-cost sensitivity
turnover becomes expensive
```

Risk:

```text
intervention may create unrealistic liquidity-blind actions
```

## 15.10 Execution / Metaorder Artifact

Signals:

```text
same-sign Δweights over multiple days
action clipping
high cumulative turnover
slow unwind
```

Primitive signature:

```text
persistent small trades
sector-level unwind
```

Risk:

```text
not a strategy primitive, but an execution artifact
```

---

# 16. Risk Register and Mitigations

| Risk | Why it may happen | Prevention |
|---|---|---|
| Codebook is uninterpretable | VQ-VAE discrete code does not guarantee semantic meaning; latent variables are non-unique | K-ablation, random codebook, shuffled labels, behavior fidelity, cross-fold stability |
| Primitive is persistence-only | Previous position / previous code explains most behavior | Previous-code baseline, Markov baseline, action-fidelity lift requirement |
| Primitive is market-regime artifact | Hidden states cluster bull/bear regimes, not strategy | HMM baseline, regime-conditioned labels, sector/regime stratification |
| Labeling is multiple-testing artifact | Many primitives × many signals | Bonferroni/FDR, 3-of-5 rule, temporal holdout, deflated Sharpe |
| Bad primitive is actually protective | Risk-off activates during bad markets | Compare raw return vs factor-adjusted and drawdown-protective contribution |
| Hidden intervention goes OOD | Modified latent leaves natural manifold | Mahalanobis/kNN/AE/local PCA OOD gates, OOD-normalized schedules |
| Intervention changes action unpredictably | Policy head nonlinear, hidden state entangled | subspace interventions, logit baseline, input-level baseline |
| Sequential intervention confounded by natural recovery | Bad episodes often decay naturally | no-intervention, shifted-window, matched-window controls |
| Decay kernel works only because dose is smaller | Exponential applies less total intervention | peak-, dose-, and OOD-normalized comparison |
| LLM/news signals leak future info | Bad timestamp handling | strict publication-time filtering, after-close rules |
| DJ30 too small | Cross-sectional labels noisy | expand to S&P 500, use sector ETFs, or mark Method 5 low confidence |
| Backtest overfit | Many interventions and schedules | walk-forward, purged CV, embargo, DSR, random schedule controls |
| Adapter overfits primitive penalty | It learns to hide primitive, not improve policy | KL to original, OOD penalty, random-label adapter baseline |
| World model hallucination | Generated markets not reliable | only use as stress-test, not primary OOS evidence |
| Causal claim too strong | Interventions may not identify true causal variables | use descriptive / predictive / causal status labels and gate claims |

---

# 17. Codex Implementation Plan

## 17.1 Suggested Repository Structure

Target structure:

```text
project/
  configs/
    stage0_ppo.yaml
    stage1_vqvae.yaml
    stage2_diagnostics.yaml
    stage3_labeling.yaml
    stage4_outcomes.yaml
    stage5_causal_audit.yaml
    stage55_sequential.yaml
    stage6_adapter.yaml

  src/
    data/
      load_finrl.py
      feature_engineering.py
      news_events.py
      splits.py

    ppo/
      train_or_load.py
      extract_rollouts.py
      policy_hooks.py
      ppo_diagnostics.py

    ssl/
      market_encoder.py
      vqvae.py
      transformer_vqvae.py
      codebook_metrics.py
      primitive_assignment.py

    diagnostics/
      portfolio.py
      action_fidelity.py
      ppo_by_primitive.py
      risk_metrics.py
      ood.py

    labeling/
      rbsa.py
      signal_correlations.py
      probes_tcav.py
      hmm_regimes.py
      synthetic_strategies.py
      mechanism_bank.py
      label_aggregation.py

    causal/
      directions.py
      subspaces.py
      one_step_interventions.py
      sequential_decay.py
      kernels.py
      gates.py
      controls.py

    adapter/
      primitive_adapter.py
      adapter_losses.py
      train_adapter.py
      eval_adapter.py

    evaluation/
      walk_forward.py
      purged_cv.py
      bootstrap.py
      deflated_sharpe.py
      tables.py
      plots.py

  artifacts/
    stage0/
    stage1/
    stage2/
    stage3/
    stage4/
    stage5/
    stage55/
    stage6/

  notebooks/
    inspect_primitives.ipynb
    mechanism_review.ipynb
    stage55_results.ipynb

  reports/
    primitive_table.md
    causal_audit.md
    final_methodology.md
```

## 17.2 Current Workspace Migration Plan

Do not create this structure immediately while Stage 0 training is running. When migration starts, map the current workspace as follows:

```text
stage0_audit/
  -> initially remains the active Stage 0 implementation

stage0_audit/feature_sets/
  -> later maps to artifacts/stage0/feature_sets/ and src/data/

stage0_audit/model_runs/
  -> later maps to artifacts/stage0/model_runs/

latent_actions_experiments/
  -> remains legacy/reference-only, not part of the active pipeline

PROPOSAL_ALIGNED_INTERPRETABILITY_RESEARCH_PLAN.md
  -> high-level proposal-aligned plan

METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md
  -> canonical detailed methodology and implementation contract
```

Pragmatic migration rule:

```text
Keep Stage 0 stable first.
After frozen model selection, create src/ and artifacts/ wrappers around existing Stage 0 outputs rather than moving files prematurely.
```

## 17.3 Main Pipeline Runner

Target commands:

```bash
python -m src.ppo.extract_rollouts --config configs/stage0_ppo.yaml
python -m src.ssl.train_vqvae --config configs/stage1_vqvae.yaml
python -m src.diagnostics.run_all --config configs/stage2_diagnostics.yaml
python -m src.labeling.run_all --config configs/stage3_labeling.yaml
python -m src.diagnostics.ppo_by_primitive --config configs/stage4_outcomes.yaml
python -m src.causal.one_step_interventions --config configs/stage5_causal_audit.yaml
python -m src.causal.sequential_decay --config configs/stage55_sequential.yaml
python -m src.adapter.train_adapter --config configs/stage6_adapter.yaml
```

Stage 6 command should internally refuse to run unless Stage 5 and Stage 5.5 gates pass.

---

# 18. Required Output Tables

## 18.1 Primitive Headline Table

```text
primitive_id
usage_rate
median_dwell
label
label_confidence
market_mechanism
portfolio_profile
sector_profile
regime_affinity
PPO_signature
P&L_contribution
risk_contribution
status
```

Status:

```text
DESCRIPTIVE_ONLY
PREDICTIVE
PPO_MECHANISTIC
CAUSAL_CANDIDATE
CONTROLLABLE_SAFE
UNSAFE_OOD
EXECUTION_ARTIFACT
REGIME_ARTIFACT
```

## 18.2 Stage 5.5 Intervention Table

```text
primitive_id
mechanism
window_length
kernel
normalization
η0
dose
OOD_max
ΔAUC_bad
Δturnover
Δdrawdown
Δpolicy_KL
Δreturn_5d
Δreturn_10d
safe_suppression_score
passed
failure_reason
```

## 18.3 Risk / Failure Table

```text
primitive_id
failure_type
evidence
recommended_next_action
```

Examples:

```text
hidden_direction_entangled
OOD_failure
no_action_fidelity
regime_artifact
protective_risk_off
execution_artifact
```

---

# 19. Minimal Viable Implementation Order

If time is limited, implement in this order:

```text
1. Stage 0 full PPO logging
2. Stage 1 VQ-VAE hidden-only + windowed transformer VQ-VAE
3. Stage 2 action fidelity + portfolio diagnostics
4. Stage 3 original five finance labels + mechanism bank
5. Stage 4 PPO diagnostics by primitive
6. Stage 5 one-step hidden/subspace interventions with OOD gates
7. Stage 5.5 exponential / power-law / reverse-decay / random-dose schedules
8. Only then consider Stage 6 adapter
```

Skip or postpone:

```text
LLM news encoder
JEPA market world model
ABIDES simulation
full adapter training
```

These are valuable extensions, but not required for the first complete paper-quality result.

---

# 20. Expected Final Narrative

The final paper / report should be able to say:

```text
We discover discrete behavior primitives in a PPO trading agent using self-supervised VQ-based representation learning.

We label primitives with finance-grounded and market-mechanism-grounded evidence.

We distinguish primitives that merely describe market regimes from primitives that predict policy outputs and portfolio behavior.

We audit PPO internals by primitive: advantage, value error, entropy, logprob, KL, clip fraction.

We test causal controllability using gated one-step and sequential interventions.

We introduce decay-gated Stage 5.5 interventions inspired by market shock-response, transient impact, and self-exciting event dynamics.

We report which primitives are descriptive only, which are predictive, which are PPO-mechanistic, and which are safe causal candidates.
```

Most important final sentence:

> The project's contribution is not "all hidden primitives are safe controls." The contribution is a methodology for separating descriptive behavior labels from predictive, PPO-mechanistic, and causally controllable market-mechanism primitives.

---

# 21. Codex Addenda: Three Methodology Improvements

These additions are intentionally limited to three. They do not change the core methodology; they make it safer to implement from the current Stage 0 codebase.

## 21.1 Stage 0 Must Produce a Versioned Export Contract

Before Stage 1, create a machine-readable export manifest next to every trajectory export:

```text
artifacts/stage0/export_manifest.json
```

Required fields:

```json
{
  "model_id": "...",
  "model_path": "...",
  "model_sha256": "...",
  "feature_set_id": "...",
  "feature_file_path": "...",
  "feature_file_sha256": "...",
  "split_policy": "...",
  "train_start": "...",
  "train_end": "...",
  "validation_start": "...",
  "validation_end": "...",
  "test_start": "...",
  "test_end": "...",
  "observation_dim": 0,
  "action_dim": 29,
  "ticker_universe": [],
  "hidden_layer_contract": {
    "policy_branch": true,
    "value_branch": true,
    "layers": []
  },
  "turbulence_threshold_active_in_training": false,
  "turbulence_threshold_active_in_eval": false,
  "transaction_cost_policy": "...",
  "created_at": "..."
}
```

Reason:

```text
Hidden-state research fails quickly if model, feature set, split, and tensor-shape provenance are ambiguous.
```

This directly addresses the current project risk: several old experiments use similar filenames but different feature sets, threshold settings, and evaluation periods.

## 21.2 Separate PPO-Input Features from Label-Only and Future-Optional Features

Every feature must be assigned one of three roles:

```text
PPO_INPUT
LABEL_ONLY
FUTURE_EXTENSION
```

Examples:

```text
technical indicators used by PPO → PPO_INPUT
VIX/regime probabilities if included in observation → PPO_INPUT
sector labels / mechanism proxies for post-hoc labels → LABEL_ONLY
news embeddings not available in Stage 0 → FUTURE_EXTENSION
```

Each feature should also include:

```text
source
publication / availability timestamp
lag policy
whether transformation was fit on train only
whether it is allowed in frozen-test selection
```

Reason:

```text
Stage 3 labels and Stage 5 intervention gates may use richer context than the PPO saw, but those features must not silently become PPO inputs or leak future information.
```

This is especially important when adding news/event/LLM context and mechanism-bank proxies.

## 21.3 Maintain a Primitive Claims Ledger

For every primitive and every stage, save an evidence ledger:

```text
artifacts/claims/primitive_claims_ledger.parquet
```

Minimum columns:

```text
primitive_id
claim_level
claim_text
stage
evidence_metric
evidence_value
split_used
control_used
passed
failure_reason
can_upgrade_to_next_level
```

Allowed claim levels:

```text
DESCRIPTIVE
PREDICTIVE
PPO_MECHANISTIC
CAUSAL_CANDIDATE
CONTROLLABLE_SAFE
```

Upgrade rule:

```text
A primitive can only move to the next claim level if it passes the required gates on a holdout or cross-period split not used to discover that same claim.
```

Reason:

```text
The main scientific danger is narrative drift: a cluster discovered descriptively can be accidentally described as causal because later plots look intuitive.
```

The ledger makes the final paper/report auditable and prevents overclaiming.

---

# 22. Current Stage 0 / Stage 0.1 Implications

Given the current project state, Codex should treat the selected Stage 0 teacher as an exploratory frozen teacher, not as the final mechanically ideal policy. The immediate workflow is:

```text
1. Preserve the selected Stage 0 teacher and Joseph export package.
2. Start Stage 1 primitive discovery on the current teacher for exploratory analysis.
3. In parallel, implement Stage 0.1 stabilized teacher experiments.
4. Use a weight-based portfolio environment with 29 stocks + cash.
5. Compare flat Dirichlet and hierarchical Dirichlet policies.
6. Compare P and PD execution controllers first; expand to PI/PID only if stable.
7. Add train-only normalization, risk/turnover-aware reward, target_kl, and conservative PPO updates.
8. Log per-timestep PPO, action, controller, and portfolio diagnostics.
9. Select the Stage 0.1 teacher by walk-forward validation only.
10. Use frozen test exactly once.
11. Compare Stage 1 primitives from the original teacher versus the stabilized teacher.
```

Repository restructuring has already started. New code should be added under `src/` and `configs/` where practical, while `stage0_audit/` remains the historical Stage 0 audit and export record.

---

# 23. Stage 0.1 Rescue Hypothesis Ledger

This section synchronizes the methodology with the Stage 0.1 results, the hidden-state comparison, the frozen diagnostic audit, and the DeepSearch reports now stored locally:

```text
reports/deepsearch/deep-research-report-12-stabilized-ppo-teacher.md
reports/deepsearch/deep-research-report-13-hierarchical-dirichlet-rescue.md
reports/STAGE0_1_ROOT_SPLIT_EXPERIMENTS_2_3_PLAN.md
artifacts/stage0_1/analysis/hierarchical_rescue_audit/STAGE0_1_HIERARCHICAL_RESCUE_AUDIT.md
artifacts/stage0_1/hidden_state_package/HIDDEN_STATE_COMPARISON_REPORT.md
artifacts/stage0_1/frozen_diagnostic/FROZEN_DIAGNOSTIC_AUDIT.md
```

The current empirical interpretation is:

```text
Stage 0.1 solved much of the mechanical PPO instability problem.
Flat Dirichlet + PD is currently the strongest walk-forward validation baseline.
Hierarchical Dirichlet variants are more cash-heavy and weaker on validation.
Hierarchical Dirichlet can look better on the 2022 frozen diagnostic because cash is protective in stress.
Hidden-state geometry suggests hierarchy is expressive, not simply too weak.
Therefore the likely failure is wrong decision factorization, not "hierarchy is impossible."
```

This section is binding for future Stage 0.1 experiments:

```text
Do not pick a teacher by frozen-test stress behavior.
Use frozen diagnostics to understand failure modes only.
Select by pre-declared walk-forward validation rule.
Change one architectural axis at a time whenever possible.
```

## 23.1 Current Stage 0.1 Empirical Baseline

The active Stage 0.1 baseline observations are:

```text
Best current validation family:
  flat_dirichlet_pd / flat_dirichlet_pid

Current hierarchy failure mode:
  hierarchical_dirichlet_* holds materially more cash,
  often around the high-teens in mean cash weight,
  which helps in stress but hurts average validation capital deployment.

Frozen diagnostic:
  useful for stress interpretation only,
  not a final model-selection criterion.

Hidden-state comparison:
  hierarchical policies show richer latent geometry than flat policies,
  so the rescue path should focus on better factorization and execution,
  not on assuming hidden states are uninformative.
```

The phrase "richer latent geometry" means:

```text
the hidden states vary across more independent directions,
are less dominated by one principal component,
and carry more separable structure about portfolio behavior.
```

This is a useful interpretability signal, but not sufficient for model selection.

## 23.2 Hypothesis Table

| ID | Idea | Problem Addressed | Hypothesis | First Test |
|---|---|---|---|---|
| H1 | Increase policy size | Stage 0.1 observation dimension is much larger than Stage 0, but the policy encoder is narrower. | `[256,128,64]` or `[512,256,128,64]` preserves more information while keeping the final 64-d latent for Stage 1. | Flat and hierarchical PD with Stage0-style width. |
| H2 | Compact feature ablation | 62 per-ticker features may add noise or redundant state. | A compact Stage0-like feature set may improve stability or reveal that high-dimensional features are hurting hierarchy. | `flat_dirichlet_pd_compact22` and later risk-root compact variant. |
| H3 | Root split cash vs invested | Cash currently competes with sectors as if it were another sector. | Separate `q_t = invested_fraction` makes cash a state-dependent risk throttle rather than a constant shortcut. | `root_split_beta_dirichlet_kp_riskcash_v1`. |
| H4 | Feature routing / separate encoders | If root sees every raw company feature, cash can become a stock-selection shortcut. | Root should receive risk/macro/portfolio summaries; risky head receives stock/sector features plus a small broadcast risk context. | Root split with separate root-risk encoder and risky allocation encoder. |
| H5 | Risk-conditioned cash prior | Constant cash penalty is wrong because cash is useful in stress. | Penalize only excessive cash relative to a risk-conditioned allowance. | Add after root split baseline, not before. |
| H6 | Dirichlet-tree root | Independent Dirichlet factors may be too rigid for branch dependencies. | Dirichlet-tree root preserves hierarchy while allowing subtree-dependent uncertainty. | Use after root split baseline is healthy. |
| H7 | Logistic-normal root/group layer | Dirichlet can be too limited on covariance over simplex. | Logistic-normal at root/group level may model cash/group dependence better while keeping Dirichlet leaves. | Use after Dirichlet-tree or as a parallel distribution upgrade. |
| H8 | Semi-static discovered clusters | Static sectors may not match true market or policy organization. | Groups discovered from residual correlations, target-weight co-movement, or hidden-state similarity improve allocation and interpretability. | Monthly or quarterly candidate groups with stability gates. |
| H9 | Cascade controllers | Root and leaf decisions live on different time scales. | Root exposure should move slowly; leaf allocation can rebalance faster inside the invested book. | Separate root/leaf Kp and dead-zone. |
| H10 | Learned bounded Kp gates | Fixed controller coefficients are arbitrary. | Policy can emit bounded rebalance speed conditioned on risk, liquidity, drawdown, and gap-to-target. | Kp-only learned gates; no learned Ki/Kd initially. |
| H11 | Asymmetric cascade | De-risking and re-risking are not symmetric. | Reduce risk faster during stress, but redeploy cash more slowly after stress. | Separate `k_derisk` and `k_rerisk` bounds. |
| H12 | No-trade / dead-zone bands | Tiny target changes can create unnecessary turnover. | Trade only when target-executed gap exceeds a risk/liquidity-dependent threshold. | Add to cascade after fixed Kp baseline. |
| H13 | Bottom-up inhibitory feedback | Lower layers observe concentration, turnover, and OOD risks that root may not see. | Lower layer may only veto, slow down, or increase cash warning; it must not freely choose root allocation. | Deterministic monotonic feedback after root split works. |
| H14 | Safety projection / QP layer | A learned feedback layer can become an opaque second policy. | Minimal action correction into a safe set is more auditable than free feedback. | Projection layer after deterministic feedback baseline. |
| H15 | CAOSD / constrained simplex decomposition | Complex group constraints need correct simplex factorization and log-prob handling. | Decomposing constrained simplex action spaces can improve PPO consistency. | Later, after root split and distribution baselines. |
| H16 | Meta-policy over interpretable base policies | If one hierarchy cannot handle risk-off, risk-on, and rotation styles, a style bank may be better. | Meta-policy selects among interpretable base policies by regime. | Fallback only if hierarchy rescue fails. |

## 23.3 Ranked Implementation Priority

| Rank | Idea | Expected Upside | Main Risk | Complexity | Priority Decision |
|---:|---|---|---|---|---|
| 1 | Root split cash vs invested | Very high | Can still learn constant cash | Medium | Run immediately. |
| 2 | Increase policy size | High | More overfitting / slower jobs | Low | Run as width control. |
| 3 | Compact feature ablation | High diagnostic value | May underfit useful features | Low | Run as confound control. |
| 4 | Feature routing / separate encoders | High | Too much hand-designed separation | Medium | Pair with root split. |
| 5 | Asymmetric cascade | High | Bad bounds may slow deployment | Medium | Add after root split baseline. |
| 6 | No-trade / dead-zone | Medium-high | Can under-trade | Low-medium | Add with cascade experiments. |
| 7 | Learned bounded Kp | Medium-high | Gate saturation / opacity | Medium | Kp-only after fixed Kp. |
| 8 | Semi-static discovered clusters | High | Cluster churn and overfitting | High | Run after root split controls. |
| 9 | Dirichlet-tree root | Medium-high | Implementation complexity | High | Distribution upgrade after baseline. |
| 10 | Logistic-normal root | Medium-high | Less interpretable density | High | Distribution upgrade / parallel test. |
| 11 | Risk-conditioned cash prior | Medium | Can encode arbitrary bias | Medium | Add only after cash behavior baseline is measured. |
| 12 | Bottom-up inhibitory feedback | Medium | Can become hidden policy correction | Medium-high | Only as safety veto after root split works. |
| 13 | Safety projection / QP layer | Medium | Engineering complexity | High | Better long-run alternative to free feedback. |
| 14 | CAOSD | Medium | Complex PPO distribution accounting | High | Later constraint-aware extension. |
| 15 | Meta-policy over styles | High fallback | Can become another black box | High | Plan B if hierarchy rescue fails. |

Do not combine many high-complexity ideas in the same first run. If a model changes width, root factorization, distribution family, clustering, and feedback simultaneously, the result is not interpretable as an experiment.

## 23.4 Root Split Architecture Contract

The preferred first hierarchy rescue architecture is:

```text
risk_features_t
    -> root_risk_encoder
    -> Beta or 2D Dirichlet root
    -> q_t = invested_fraction

stock/group_features_t + broadcast(z_risk)
    -> risky_encoder
    -> Dirichlet or hierarchical risky allocator
    -> u_t = allocation over risky assets or groups

final target:
    w_cash,t = 1 - q_t
    w_stock_i,t = q_t * u_i,t
```

For sector/group hierarchy:

```text
q_t                 = invested fraction
g_{s,t}             = group/sector share of invested capital
u_{i|s,t}           = within-group stock share
w_{i,t}             = q_t * g_{s(i),t} * u_{i|s(i),t}
w_cash,t            = 1 - q_t
```

PPO must store the hierarchical action factors, not only the final weights:

```text
root action:
  q_t
  log_prob_root
  root_entropy
  root_concentration

group action:
  g_t
  log_prob_group
  group_entropy

leaf action:
  u_{i|g,t}
  log_prob_leaf
  leaf_entropy

total log_prob:
  log_prob_root + log_prob_group + sum(log_prob_leaf)
```

The final weight vector is an environment-facing deterministic mapping from the sampled factors. This avoids ambiguous PPO diagnostics.

## 23.5 Feature Routing Rule

Root and risky heads should not receive identical unrestricted raw inputs by default.

Root/cash-risk head should emphasize:

```text
VIX / volatility / realized volatility
market and portfolio drawdown
market trend
market breadth
correlation spike
liquidity stress summaries
macro deltas
previous cash and invested fraction
recent turnover and target-to-executed gap
cross-sectional summaries, not raw stock tables
```

Risky allocation head should emphasize:

```text
per-stock returns and momentum
per-stock volatility and drawdown
price/SMA distance
volume/liquidity proxies
technical and fundamental deltas
sector/group embedding
relative ranks inside the universe
small broadcast risk embedding from the root encoder
```

Rationale:

```text
root = market exposure / cash / risk appetite
risky = asset selection / group allocation / stock allocation
```

If root receives all company-level features without constraint, cash can become a shortcut for stock selection. If risky allocation receives no market context, it may select stocks similarly in calm and crash regimes. The design must separate decision type, not fully isolate information.

## 23.6 Dynamic Discovered Hierarchy Rule

Dynamic regrouping must be semi-static and causal.

Default cadence:

```text
candidate groups:
  recompute monthly using only past data

applied groups:
  update monthly or quarterly only if stability gates pass

rolling window:
  start with 252 trading days;
  test 504 trading days as a smoother alternative
```

Crisis or regime shifts should trigger re-evaluation, not automatic group replacement:

```text
if VIX / realized vol / drawdown / correlation regime changes sharply:
  allow off-cycle candidate-group check
  apply only if groups are stable and useful
```

Group construction candidates:

```text
residual-correlation communities
rolling return-correlation communities
target-weight co-movement communities
executed-weight co-movement communities
hidden-state-to-asset exposure similarity
sector-prior plus controlled deviations
```

Group update gates:

```text
no look-ahead
cluster stability over 2-3 windows
reasonable group-size floor
group churn below threshold
within-group co-movement improves
turnover from regrouping is bounded
validation behavior improves over sector prior
labels remain explainable
```

Do not recluster daily. Daily cluster changes are likely to encode noise and will damage both training stability and interpretability.

## 23.7 Cascade and Feedback Control Contract

The cascade design is:

```text
root controller:
  changes invested fraction / group budgets
  slower by default
  larger dead-zone
  stronger turnover penalty

leaf controller:
  changes stock mix inside already allocated capital
  faster by default
  smaller dead-zone
```

Example bounded update:

```text
q_exec,t = q_exec,t-1 + k_root,t  * (q_target,t - q_exec,t-1)
u_exec,t = u_exec,t-1 + k_inner,t * (u_target,t - u_exec,t-1)
```

Asymmetry is required:

```text
if q_target,t < q_exec,t-1:
  use k_derisk,t    # faster risk reduction
else:
  use k_rerisk,t    # slower redeployment after stress
```

Root-to-leaf modulation is allowed:

```text
risk-off root state -> lower inner trading speed
high root uncertainty -> wider leaf dead-zone
```

Bottom-up feedback is allowed only as inhibitory safety feedback:

```text
lower layer may:
  reduce a sector/group effective target
  increase global cash warning
  slow leaf execution
  widen dead-zone

lower layer may not:
  freely choose a new root allocation
  create new positive bets outside the root intent
  become an unlogged second policy
```

Two-channel deterministic feedback is the preferred first version:

```text
sector-specific veto:
  g_eff_s proportional to g_raw_s * exp(-lambda_sector * feedback_s)
  then renormalize inside invested capital

global cash veto:
  F_global = sum_s g_raw_s * feedback_s
  q_eff = q_raw * (1 - lambda_global * F_global)
```

Feedback risk metrics:

```text
within-group HHI
max stock weight inside group
turnover needed to reach target
target-to-executed gap
liquidity stress
volatility stress
inner OOD distance
simplex projection residual
```

Pass criteria:

```text
feedback active sometimes, not always
feedback correlates with concentration / turnover / liquidity / OOD risk
feedback lowers drawdown or tail loss without destroying calm-regime return
raw-to-effective gap does not dominate behavior
hidden primitives do not become merely "veto layer active" labels
```

Kill criteria:

```text
feedback active on most days
raw action is usually unsafe
raw-to-effective gap dominates target-to-executed behavior
cash returns to constant high levels
validation score falls below the static hierarchy without compensating risk benefit
```

Longer-run alternative:

```text
replace free feedback with a safety projection / QP layer that minimally projects raw action into a safe feasible set.
```

This is more auditable but more complex.

## 23.8 Experiment Order

The next experiments should be organized as batches.

### Batch A: Encoder and Feature Confound Controls

```text
flat_dirichlet_pd_stage0style256_v1
hierarchical_dirichlet_pd_stage0style256_v1
flat_dirichlet_pd_compact22_v1
```

Purpose:

```text
separate architecture failure from encoder-width or feature-noise failure.
```

### Batch B: Minimal Root Split

```text
E2a_root_split_beta_dirichlet_pd_nocashprior_v1
E2b_root_split_beta_dirichlet_pd_riskcash_v1
E3_root_split_beta_dirichlet_pd_separate_encoders_v1
```

Purpose:

```text
test whether cash becomes state-dependent rather than constant high.
```

Implementation rule:

```text
Keep the current final-weight PD execution first, so the root-split pair tests
action decomposition and feature routing before changing the controller.

Factor-level Kp, asymmetric speed, dead-zone, and feedback are later execution
experiments, not part of the minimal root-split causal pair.
```

### Batch C: Execution Improvements

```text
root_split_beta_dirichlet_asym_speed_v1
root_split_beta_dirichlet_deadzone_v1
root_split_beta_dirichlet_learned_kp_v1
```

Purpose:

```text
test slow root / fast leaf, de-risk faster than re-risk, and bounded learned Kp.
```

### Batch D: Distribution Family Upgrades

```text
hierarchical_riskcash_dirtree_root_v1
hierarchical_riskcash_logitnormal_root_v1
```

Purpose:

```text
test whether independent Dirichlet factors are the bottleneck.
```

### Batch E: Discovered Hierarchy

```text
hierarchical_discovered_rescorr_kp_v1
hierarchical_discovered_targetcov_kp_v1
hierarchical_discovered_hiddenstate_kp_v1
```

Purpose:

```text
test whether static sectors are the wrong grouping.
```

### Batch F: Safety Feedback and Constraint-Aware Extensions

```text
root_split_deterministic_bottomup_veto_v1
root_split_projection_safety_layer_v1
caosd_constrained_simplex_v1
```

Purpose:

```text
test safety veto / action projection after the core root split is already healthy.
```

### Batch G: Fallback Strategy Bank

```text
meta_policy_interpretable_styles_v1
```

Purpose:

```text
fallback if one monolithic hierarchy cannot support both offensive allocation and defensive risk-off behavior.
```

Minimum next five if compute is limited:

```text
1. flat_dirichlet_pd_stage0style256_v1
2. hierarchical_dirichlet_pd_stage0style256_v1
3. flat_dirichlet_pd_compact22_v1
4. root_split_beta_dirichlet_kp_riskcash_v1
5. root_split_beta_dirichlet_kp_separate_encoders_v1
```

## 23.9 Additional Logging Required for These Hypotheses

Future Stage 0.1 variants must log:

```text
q_target
q_effective
q_executed
cash_target
cash_effective
cash_executed
root_alpha_invested
root_alpha_cash
root_concentration
root_entropy
root_log_prob
risky_entropy
group_entropy
within_group_entropy
k_root
k_derisk
k_rerisk
k_inner
dead_zone_root
dead_zone_inner
raw_group_weights
effective_group_weights
raw_stock_weights
effective_stock_weights
feedback_s
F_global
cash_change_due_to_feedback
sector_change_due_to_feedback
raw_to_effective_gap
effective_to_executed_gap
raw_to_executed_gap
group_cluster_version
group_cluster_source
group_churn
within_group_correlation
hidden_effective_rank
hidden_temporal_delta
CKA_to_baseline
```

The main root-split diagnostic is:

```text
cash should be low in calm/risk-on windows and higher in high-volatility/drawdown windows.
```

The main feedback diagnostic is:

```text
feedback should be a rare, interpretable safety veto, not the main policy.
```

The main discovered-hierarchy diagnostic is:

```text
groups should be stable enough to label and useful enough to improve validation or reduce turnover/risk.
```

---

# References

[1]: https://arxiv.org/abs/1707.06347 "Proximal Policy Optimization Algorithms"
[2]: https://arxiv.org/abs/1711.00937 "Neural Discrete Representation Learning"
[3]: https://arxiv.org/abs/2503.14973 "Behaviour Discovery and Attribution for Explainable Reinforcement Learning"
[4]: https://arxiv.org/abs/1909.03368 "Designing and Interpreting Probes with Control Tasks"
[5]: https://arxiv.org/abs/2106.02997 "Causal Abstractions of Neural Networks"
[6]: https://arxiv.org/abs/2007.04612 "Concept Bottleneck Models"
[7]: https://arxiv.org/abs/1602.02735 "Linear models for the impact of order flow on prices I. Propagators: Transient vs. History Dependent Impact"
[8]: https://arxiv.org/abs/1502.04592 "Hawkes processes in finance"
[9]: https://arxiv.org/abs/1904.12066 "ABIDES: Towards High-Fidelity Market Simulation for AI Research"
[10]: https://arxiv.org/abs/2106.10466 "TS2Vec: Towards Universal Representation of Time Series"
[11]: https://arxiv.org/abs/2301.08243 "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
[12]: https://arxiv.org/abs/1908.10063 "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
[13]: https://en.wikipedia.org/wiki/Purged_cross-validation "Purged cross-validation"
[14]: https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio "Deflated Sharpe ratio"
[15]: https://stable-baselines3.readthedocs.io/en/v2.0.0/guide/rl_tips.html "Stable-Baselines3 RL Tips and Tricks"
[16]: https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html "Stable-Baselines3 PPO documentation"
[17]: https://finrl.readthedocs.io/en/latest/tutorial/Introduction/PortfolioAllocation.html "FinRL Portfolio Allocation"
[18]: https://www.mdpi.com/2075-1680/11/12/664 "A Selective Portfolio Management Algorithm with Off-Policy Reinforcement Learning Using Dirichlet Distribution"
[19]: https://www.microsoft.com/en-us/research/publication/dirichlet-tree-distribution/ "The Dirichlet-tree Distribution"
[20]: https://www.investor.gov/introduction-investing/investing-basics/glossary/form-13f-reports-filed-institutional-investment "Form 13F - Reports Filed by Institutional Investment Managers"
[21]: https://arxiv.org/abs/2010.01687 "Learning Risk Preferences from Investment Portfolios Using Inverse Optimization"
[22]: https://arxiv.org/abs/2201.08445 "A Prescriptive Dirichlet Power Allocation Policy with Deep Reinforcement Learning"
[23]: https://arxiv.org/abs/2011.05381 "Dirichlet Policies for Reinforced Factor Portfolios"
[24]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 "Building Diversified Portfolios that Outperform Out of Sample"
[25]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540 "The Hierarchical Equal Risk Contribution Portfolio"
[26]: https://arxiv.org/abs/1801.08757 "A Simple Solution for Safe Exploration in Continuous Action Spaces"
[27]: https://proceedings.mlr.press/v70/achiam17a.html "Constrained Policy Optimization"
[28]: https://arxiv.org/abs/1703.01161 "FeUdal Networks for Hierarchical Reinforcement Learning"
[29]: https://arxiv.org/abs/1609.05140 "The Option-Critic Architecture"
[30]: https://arxiv.org/abs/2105.08664 "Deep Graph Convolutional Reinforcement Learning for Financial Portfolio Management"
[31]: https://arxiv.org/abs/2407.15532 "Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks"
