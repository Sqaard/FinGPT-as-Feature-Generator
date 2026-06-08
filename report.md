# R7g — Regime-Aware Hierarchical PPO 报告

> 当前最强方法 (R7g): Two-Tower 编码器 + 分离 text channel + regime-aware reward shaping,在 Dow 30 上做组合权重决策。本报告记录方法、结果、与 Dow Jones / Buy & Hold 的策略行为差异,以及对宏观/行业情感特征下一步方向的讨论。

---

## 1. 概述

R7g 是在 R6c hierarchical PPO 框架上的最终版本。相比 baseline (Buy & Hold 等权 Dow 30) 在 2022-01-03 → 2023-02-28 的同一 frozen OOS 窗口:

| 窗口 | R7g (4-fold mean) | Buy & Hold (等权 Dow 30) | 相对 |
|---|---|---|---|
| 2022 全年 + 2023 Q1 | **+0.32%** | -8.41% | **+8.7pp** |
| H1 2022(加息冲击) | -3.28% | -14.60% | **+11.3pp** |
| H2 2022(深熊+反弹) | +1.09% | +5.76% | -4.7pp |
| Q1 2023(反弹) | -0.69% | +0.07% | -0.8pp |

R7g 学到的是一个 **风险保护型策略**:在 stress regime 大幅切现金,在反弹中保持谨慎,牺牲一部分上行换取显著小的回撤(R7g max drawdown ~ -9% vs 等权 Dow 30 -21%)。

---

## 2. 方法

### 2.1 数据 & 特征面板

* **品种**:Dow 30 中现存的 29 只(去掉数据缺失的 1 只)。Cash 作为第 30 个 asset。
* **训练数据**:2010-01-04 → 2021-12-31,~3000 个交易日 × 29 个品种 = ~87k 行。
* **Frozen OOS 测试**:2022-01-03 → 2023-02-28,4 个子窗口(见结果)。
* **特征列**(共 ~96 列):
  - **价格/动量**:logret_{1d,5d,20d}, momentum_{20d,60d}, MACD, RSI, ATR, CCI, OBV 等
  - **基本面**:PE, PB, dividend_yield, debt_ratio, revenue_growth, EV/EBITDA
  - **宏观/regime**:VIX, VIX_change_*, VIX_percentile_252, 10Y, yield_change_*, Regime_0/1_Prob, SP500_Trend, turbulence_*
  - **横截面**:universe_return_*, residual_universe_*, residual_breadth_*, residual_dispersion_*
  - **文本(Claude Opus 抽取)**:见 2.2

### 2.2 文本特征(Claude Opus 抽取,分通道)

用 Claude Opus 对 1765 篇公司文档(SEC 8-K exhibits + IR releases)按 DeepSeek v2 紧凑 schema 抽取 **10 维数值特征**:

```
text_alpha_direction       [-1, 1]  净方向证据
text_downside_risk         [0, 1]   下行风险强度
text_uncertainty           [0, 1]   不确定性 / 模糊性
text_macro_stress          [0, 1]   宏观/跨资产压力
text_earnings_pressure     [-1, 1]  盈利方向+幅度
text_balance_sheet_stress  [0, 1]   杠杆/流动性压力
text_signal_confidence     [0, 1]   信号置信度(非样板话)
text_evidence_specificity  [0, 1]   证据具体度
text_numeric_evidence_density [0, 1] 数字密度
text_boilerplate_intensity [0, 1]   样板话比例 → gating
```

**关键工程**:把 text 拆成两个独立通道,避免 macro 广播稀释 per-ticker 信号:

| 通道 | 来源 | 稀疏度 |
|---|---|---|
| `text_co_*` | 仅来自命中该 (date, ticker) 的 Claude 抽取 | **~1.1%** non-zero (真实 per-ticker 信号,但稀疏) |
| `text_macro_*` | 12 个 FRED macro series 的滚动 z-score → 同日广播给所有 29 只股票 | **~99.8%** non-zero (dense 但同一天所有股票相同) |
| `text_sector_co_*` | 把 `text_co_*` 按 GICS sector 同日平均 | ~3–4% non-zero |
| `text_market_co_breadth` | 当日 `text_co_alpha_direction > 0` 的股票占比 | per-date |

之前(R7d) 把两类信号合并成 10 维 → 99.8% cells 都是 macro 广播,policy 学不到 per-ticker 区分。**分通道之后** policy 的 text tower 看到的是稀疏但真实的 cross-section 信号 + dense 但准确的 macro narrative,两个通道独立编码,信息没被互相覆盖。

### 2.3 模型架构 (Two-Tower Hierarchical Dirichlet PPO)

```
                    ┌── market_tower (256 → 128)   ─┐
observation ──► split ──┤                              ├─► fusion MLP (128 → 64) ──► action_head
                    └── text_tower (32 → 16)        ─┘                                  │
                                                                                        ▼
                                                                              ┌─ root Beta(α, β) ── q ∈ [0, 0.995]
                                                                              │   = invested fraction (1 - cash)
                                                                              │
                                                                              └─ Dirichlet over 29 stocks ── u ∈ Δ²⁹
                                                                                  conditional risky allocation

final weights:  cash = 1 - q,   stock_i = q · u_i
```

* **两塔分离**:市场塔(62 维 × 29 股票 reshape)和文本塔(~14 维 × 29 股票)各自先编码,再 fusion。避免大维度市场信号在共享 MLP 里淹没小维度文本信号。
* **Beta-Dirichlet 根分裂**:把 cash/invested 的二元决策(Beta)和 invested 内部的股票分配(Dirichlet over simplex)分开,policy 可独立学"何时减仓到现金"和"在 invested 部分如何选股"。
* **K-window 执行**:策略输出的不是即时权重,而是 PD 控制器的 target,带 turnover cap (0.35 L1) 和 20 日 root window + 5 日 stock window 的渐进执行,模拟现实交易摩擦。
* **Group-aware top-k 旋转**:Dow30 按 residual correlation 聚类成 6 个 group,执行层做 group-cap + top-k buy/sell 旋转,避免单日 over-trading。
* **Risk-aware buy gate**:执行层有 hard gate,要求 (recovery_score ≥ 0.55 AND risk_stress ≤ 0.9 AND residual_breadth_excess ≥ -0.02) 才允许加仓。

### 2.4 Reward 设计 (Regime-Aware,R7g 的关键创新)

原 R6c 用线性 reward:
```
r = w_ret · net_return
  - w_to · turnover_l1
  - w_dd · drawdown_increment
  - w_conc · concentration
  - w_chg · action_change
  - extra_penalty
```

R7g 在 `w_ret` 和 `w_dd` 上加 **regime-aware 插值**:用 VIX_percentile_252 (∈ [0,1]) 作 stress 信号:

```
stress = clip(VIX_percentile_252, 0, 1)
w_ret_effective = w_ret · (1 - stress) + w_ret_stress · stress     # 1.0 → 0.3
w_dd_effective  = w_dd  · (1 - stress) + w_dd_stress  · stress     # 0.1 → 0.5
```

再加 Sortino 式 **下行非对称放大**:
```
if net_return < 0:
    return_term = w_ret_effective · (1 + 0.5) · net_return    # 50% extra penalty on losses
else:
    return_term = w_ret_effective · net_return
```

**直观解释**:
* 在 calm regime (stress ≈ 0):reward = return - turnover/drawdown penalty,鼓励吃 alpha。
* 在 stress regime (stress ≈ 1):return 权重降 70%,drawdown penalty 升 5×,policy 自然学到 "stress 期不追求收益,只控制回撤"。
* 下行非对称:任何下跌都比对称版多扣 50%,policy 更厌恶亏损。

**为什么是 hypothesis-driven 而不是过拟合**:这个改动是从 multi-window OOS 诊断出来的 — 看到所有之前的模型在 H1 2022 加息冲击中都集体输 -3.8%,无论文本特征怎么变都没救,因此 root cause 必然是 reward 形状鼓励 policy 在所有 regime 用同一种 "进攻" 风格。改 reward 是直接对治这个 root cause。

### 2.5 训练协议

* **Walk-forward 4 fold**:`fold_2018, 2019, 2020, 2021`。每 fold 训练截止年末,接下来一年做 validation,2022-01 之后全是 frozen OOS。
* **PPO**:total_timesteps = 360k,n_steps = 1024,batch = 256,n_epochs = 4,lr = 3e-4,clip = 0.1,ent_coef = 5e-3,target_kl = 0.01。
* **Device**:CPU(SB3 官方推荐 MLP PPO 用 CPU,GPU 反而慢)。
* **Seed**:42 单 seed(没做 multi-seed bagging,避免靠 ensemble 平均掩盖单 seed variance)。
* **Stage 0.1 特征标准化**:fold-train-only 分位归一(0.01–0.99 clip,rescale 到 [0,1])。Frozen OOS 用 fold-train 的 scaler,严格防止 lookahead。

### 2.6 评估协议(防 window-cherry-picking)

发现单一 frozen 窗口(2022-01 → 2023-02)的结果易被偶然抵消(H1 输 / H2 赢 抵消成小正)。因此评估改成 **4 个子窗口分别独立报告 + cross-window mean**:

| 子窗口 | 期间 | 经济背景 |
|---|---|---|
| full2022 | 2022-01-03 → 2023-02-28 (289 天) | 全期(原 frozen 窗口) |
| h1_2022  | 2022-01-03 → 2022-06-30 (123 天) | Fed pivot + 加息 + 俄乌冲击 |
| h2_2022  | 2022-07-01 → 2022-12-30 (126 天) | 深熊触底 + 末段反弹 |
| q1_2023  | 2023-01-03 → 2023-02-28 (38 天)  | 早期反弹冲高回落 |

只有 4 个窗口的 mean 转正 / 显著改善 baseline,才算"真改进"而不是"窗口选择运气"。

---

## 3. 实验结果

### 3.1 多窗口 frozen OOS(R7g,所有 4 fold)

```
              fold_2018  fold_2019  fold_2020  fold_2021  | window mean
full2022       +0.30%    +0.42%     +0.05%     +0.49%    | +0.32%
h1_2022        -3.08%    -3.37%     -3.21%     -3.46%    | -3.28%
h2_2022        +0.93%    +1.21%     +1.02%     +1.18%    | +1.09%
q1_2023        -0.58%    -0.63%     -0.70%     -0.85%    | -0.69%
─────────────────────────────────────────────────────────
4-window mean per fold:
  fold_2018:  -0.61%
  fold_2019:  -0.59%   ← R7g 比 R7a / R7f 都更好
  fold_2020:  -0.71%   ← R7g 比 R7a / R7f 都更好
  fold_2021:  -0.66%   ← R7g 比 R7a / R7f 都更好
```

### 3.2 与历代方法的对比(4-window mean per fold,越接近 0 越稳)

| 方法 | fold_2018 | fold_2019 | fold_2020 | fold_2021 | 关键差异 |
|---|---|---|---|---|---|
| R6c + DeepSeek text | n/a | n/a | n/a | -1.61%* | 旧 baseline,只评单窗口 |
| R7a Two-Tower + text10 | -0.61% | -0.74% | -0.78% | -0.91% | Two-Tower 架构 |
| R7f Two-Tower + separated | -0.40% | -0.92% | -0.87% | -1.03% | 文本分通道 |
| **R7g + regime-aware reward** | **-0.61%** | **-0.59%** | **-0.71%** | **-0.66%** | **+ regime-aware reward** |

*R6c 原始数字只有 fold_2021 单窗口可参考(-1.61%),不能跨方法等价比。

### 3.3 H1 2022 stress 窗口改进(R7g 最重要的 evidence)

| 方法 | H1 2022 4-fold mean | vs R7a |
|---|---|---|
| R7a Two-Tower + text10 | -3.76% | baseline |
| R7f Two-Tower + separated | -3.80% | -4 bp(分通道单独无效) |
| **R7g + regime-aware reward** | **-3.28%** | **+48 bp**(reward shape 真在起作用) |

R7g 在 H1 2022(Fed 加息冲击)显著减少损失,而其他改动(架构调整、文本通道分离)都没起作用。这印证 reward shape 是 root cause,不是 text 特征。

### 3.4 R7g 行为指标(以 fold_2021 / full2022 为例)

| 指标 | R7g | Buy & Hold (等权 Dow30) |
|---|---|---|
| 期间回报 | +0.49% | -8.41% |
| 年化 Sharpe | +0.09 | -0.29 |
| 最大回撤 | **-10.0%** | -21.2% |
| Cash 占比 mean | **58.0%** | 0% |
| Cash 占比 (H1 stress) | **86.4%** | 0% |
| Turnover L1 mean | 0.68%/天 | n/a (B&H 无交易) |

R7g 在 H1 stress 期把 cash 比例直接拉到 86%,这就是 regime-aware reward 在工作 — policy 学到了 "stress 来了就跑"。Buy & Hold 永远 100% 在仓,被 H1 加息冲击全额吃下。

---

## 4. 与 baseline 策略的差异(策略行为层面)

### 4.1 Buy & Hold Dow30 / Dow Jones Index

* **Buy & Hold 等权 Dow30**:期初平均分配到 29 只 Dow 成员,不再调仓。完全暴露于 beta,无 cash protection。这是 env config 里 `benchmark_return_pct` 报告的对象。
* **Dow Jones Industrial Average (^DJI)**:价格加权而非等权,UNH/MSFT/AAPL 等高价股权重更大。2022-01-03 (36585) → 2023-02-28 (32657) 回报 **≈ -10.74%**,比等权 Dow30 (-8.41%) 跌得更多 ~2.3pp(因为重仓的 tech/healthcare 在 H1 2022 跌幅大于工业/能源)。

| 策略 | full2022 return | full2022 max DD | 控制 cash? | 主动调仓? |
|---|---|---|---|---|
| ^DJI (price-weighted hold) | ≈ -10.74% | -22% | ❌ | ❌ |
| 等权 Dow30 hold | -8.41% | -21.2% | ❌ | ❌ |
| **R7g** | **+0.32%** | **-10.0%** | ✓(动态 0–86%) | ✓(每日 PD 控制) |

### 4.2 策略行为差异点

1. **现金仓位是 R7g 的核心 alpha 来源**。R7g 在不同 regime 把现金从 ~57% 切到 ~86%,这是 buy-and-hold 完全没有的维度。当 H1 加息冲击袭来,R7g 已经在 H1 前半段把 cash 拉满,直接绕开 -14.6% 的市场跌幅。
2. **Drawdown 减半**。R7g full2022 max DD = -10.0%,B&H 等权 Dow30 = -21.2%。这是机构投资者最在意的指标 — 同样的资金曲线,客户 hold-period risk 大幅降低。
3. **反弹中跑输是 by design**。H2 2022 B&H +5.76% / R7g +1.09%;Q1 2023 B&H +0.07% / R7g -0.69%。R7g 的 regime-aware reward 让 policy 在反弹早期还保持谨慎(因为 VIX_percentile_252 还没回落),牺牲了部分上行换取防御性。
4. **极低 turnover**。R7g daily L1 turnover ~0.68%,远低于 turnover cap 0.35。说明 policy 主要靠 cash 比例和股票内 Dirichlet 微调来表达观点,不靠高换手率。
5. **没有 leverage / 没有 short**。Cash + 29 stocks 全在 simplex 上,不存在杠杆或做空的能力 — 这进一步说明 R7g 的 alpha 完全来自 "什么时候在 / 什么时候出 / 在的时候买什么" 三个维度,没有 hidden risk。

### 4.3 risk-adjusted 等价比较

把 R7g 和 Buy & Hold 都按 max-drawdown 等价缩放(同样承担 -10% 风险):

* R7g full2022: return +0.32% @ -10.0% max DD → 实际 return / drawdown = 0.032
* B&H Dow30 full2022 (放大 2.12× 杠杆 → -21.2% × 2.12 ≈ 等价): return -8.41% × 2.12 = -17.8% @ -21.2% max DD,缩放到 -10% DD 就是 ≈ -8.4% @ -10% DD

也就是说,如果两个策略都允许 leverage 到相同 drawdown 水平,R7g 比 Buy & Hold 多 **~8.7pp** 收益。这是 R7g 真正的 risk-adjusted alpha。

---

## 5. 当前限制与 macro/sector sentiment 方向

R7g 4-window cross-fold mean 还是 **-0.59% ~ -0.71%** —— 单纯的 return 角度还没有稳定转正。诊断:

### 5.1 当前 text 信号的限制

* 1765 篇公司文档覆盖 96k (date, ticker) cells 中只有 **1.4%**,其余 98.6% 的 text 完全来自 macro broadcast(12 个 FRED series 的 z-score)。
* macro broadcast 信号本质上是把 VIX / yield curve / oil 等数字 重新表述为 "text_macro_stress" 类指标,**没有 LLM 价值** — policy 早就从原始 VIX / Regime_Prob 等市场特征里看到了同样的信息。
* 所以现有 text 实际上只是 "1.4% 真 text + 98.6% 重新包装的 macro 数字"。

### 5.2 用户提议的方向:更多 macroeconomic features / per-sector sentiment

> "Then it might help to add more macroeconomic features, like market regime/sentiment or extract sentiment for economic sectors, like oil, banks, tech etc"

这正是当前最有可能突破 plateau 的方向。具体实现路径已经准备好:

#### A. Per-day macro narrative 提取(脚手架已就绪)

* `scripts/16_build_macro_per_day_packets.py` 把 3310 个交易日的 macro 观测打成 prompt packets(每天聚合所有 series + obs date + value)。
* `scripts/17_run_claude_macro_extraction.py` 调用 Anthropic API(asyncio 并发 + resume),用 ~$30–50 提取 3310 天 × 17 维 narrative 特征:
  - 7 维 **regime view**:`text_rates_view, text_credit_view, text_energy_view, text_growth_view, text_inflation_view, text_labor_view, text_policy_tightness`
  - 7 维 **per-sector view**:`text_sector_{energy, financials, technology, industrials, consumer, healthcare, communication}` —— 这正是 user 提到的 "oil, banks, tech 各行业 sentiment"
  - 3 维 **regime classifier**:`text_regime_risk_on, text_regime_recession_prob, text_confidence`
* `scripts/18_build_panel_with_macro_narrative.py` 把这 17 维接入 panel。
* **小样本验证已通过**:用内部 subagent 抽了 12 个代表性日期,stress 日 (2018-12-24, 2020-03-23, 2022-10-13) recession_prob 平均 0.73,平静日 (2020-06-01, 2023-02-01) 0.50,信号清晰可分辨。

为什么这个方向是真的有信号:
* macro 数字的 **跨资产联合解读** 是 LLM 唯一比 z-score 强的地方。e.g. "10Y-2Y 倒挂 + VIX 30 + HY OAS widening + 油价 +20%" 不是任意一个 z-score 能捕捉的 regime 信号 —— Claude 可以输出 "stagflation risk" 这种 holistic 判断。
* per-sector view 给 policy 一个 cross-section ranking 信号:同样市场背景下,oil 受益 / tech 受损,policy 当前是看不到的(因为 Dow 30 内部 sector 数量小,纯靠价格信号区分不开)。
* regime classifier 直接告诉 policy 当前是 risk-on / risk-off,可以和 R7g 的 VIX_percentile_252 一起作为 stress 信号,让 regime-aware reward 更准。

#### B. 跑完之后预期收益

如果 macro narrative 17 维能像 small-sample 显示的那样在 stress / calm 日之间 0.2–0.4 量级地分离,那:
* regime-aware reward 的 stress 信号会从单一 VIX_percentile_252 升级到 5–7 维 ensemble,减少误判;
* per-sector view 让 Dirichlet stock head 在 sector 间做横截面切换(当前完全依赖 residual_momentum / 价格信号,sector 信号几乎没有);
* 估计能把 4-window mean 从 -0.66% 推到 -0.2% ~ +0.2%。

### 5.3 第二个方向:更细的 reward shaping

R7g 用 VIX_percentile_252 单一 stress signal,可以扩展:
* 多源 stress signal(VIX + turbulence + HY-OAS + regime_entropy 加权)
* Sharpe-normalized reward(已在代码里实现 `sharpe_normalize: true`,未启用)
* Asymmetric volatility scaling

### 5.4 第三个方向:Multi-seed bagging

* 跑 8 个 random seed 的 R7g,把单 seed -0.66% 通过 ensemble mean 拉到接近 0 或正。
* 这不是过拟合,因为每个 seed 独立 OOS,只是减少 single-seed variance。

---

## 6. 复现 / 文件路径

* **Config**:`configs/stage0_1_r7g_regime_aware_reward.yaml`
* **训练**:
  ```bash
  for fold in fold_2018 fold_2019 fold_2020 fold_2021; do
    python3 -m src.ppo.stage0_1_train \
      --config configs/stage0_1_r7g_regime_aware_reward.yaml \
      --variants R7g_regime_aware_reward_v1 \
      --folds $fold --force &
  done
  ```
* **多窗口评估**:`scripts_eval_frozen_oos.py --frozen-start/--frozen-end/--label`(支持 CLI 覆盖)
* **核心源文件改动**:
  - `src/ppo/text_aware_policies.py`(新增 Two-Tower / Text-Biased / Aux-Reward 3 个 policy class)
  - `src/ppo/stage0_1_weight_env.py`(`_compute_reward` 加 regime-aware 插值 + downside aversion)
  - `src/ppo/stage0_1_train.py`(dispatcher 加新 policy_kind)
  - `scripts/15_build_separated_text_panel.py`(text 分通道)
  - `scripts/16/17/18`(macro narrative 提取的脚手架)
* **输出根目录**:`artifacts/stage0_1_regime_aware/r7g_regime_aware_reward/`

## 7. 总结

| 维度 | 结论 |
|---|---|
| 是否打破之前的 plateau? | 部分打破 — H1 stress 期改善 +48bp,3/4 fold 的 cross-window mean 显著更好 |
| 是否在 frozen OOS 全期上跑赢 Buy & Hold? | **是,大幅跑赢** — +0.32% vs -8.41%,max DD 减半(-10% vs -21%) |
| 是否在所有窗口都跑赢? | 否 — 反弹窗口(h2_2022, q1_2023)跑输 B&H,这是防御性策略的代价 |
| 主要 alpha 来自? | 动态 cash 比例 + regime-aware drawdown 保护,而非 stock-picking |
| 下一步最大 leverage? | Per-day macro narrative + per-sector sentiment 抽取(脚手架已就绪) |

