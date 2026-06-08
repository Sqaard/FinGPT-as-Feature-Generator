# Literature-Backed Text Features For DeepSeek Extraction

Goal: choose text features that are defensible from finance/accounting NLP
literature before running expensive PPO experiments.

## Evidence From Literature

| Literature signal | What the literature says | Feature implication |
|---|---|---|
| Finance-specific tone | General dictionaries misclassify common finance words; finance-specific positive/negative, uncertainty, litigious, modal, and constraining lists are more appropriate for 10-Ks. | Separate `financial_negative_tone`, `financial_positive_tone`, `uncertainty_modal_intensity`, `litigation_regulatory_risk`. |
| Media pessimism | High media pessimism predicts market price pressure and high trading volume, but can mean sentiment/noise rather than fundamentals. | Keep tone, but do not use it alone as alpha; combine with confidence and source controls. |
| Readability / obfuscation | More complicated annual reports are associated with lower earnings persistence for profitable firms; MD&A lexical properties are informative. | Add `readability_obfuscation_risk`, not just sentiment. |
| Risk factor disclosures | Item 1A risk-factor disclosures reflect firm-specific risks and relate to systematic risk, idiosyncratic risk, information asymmetry, and firm value. | Add `risk_factor_specificity`; generic risk text should be treated as weak signal. |
| Forward-looking MD&A | Forward-looking statement tone in MD&A is associated with future earnings; generic dictionaries can fail on corporate filings. | Add `forward_looking_guidance_direction`. |
| Earnings press-release language | Earnings-release language has information beyond concurrent numeric disclosures and is associated with future ROA and announcement-window market response. | Add `earnings_release_manager_tone` and `numeric_fundamental_evidence_density`. |
| Financial constraints | 10-K constraining-language frequency predicts liquidity events such as dividend omissions/increases, equity recycling, and underfunded pensions better than common constraint indexes. | Add `financial_constraint_liquidity_stress`. |
| Domain language models | Financial sentiment needs domain language understanding; finance-specific models outperform generic approaches on financial sentiment tasks. | DeepSeek prompt must be domain-specific and source-specific; do not ask for generic sentiment only. |

## Recommended Top Feature Set

Use this as the first DeepSeek schema to test:

| Feature | Range | Why it belongs |
|---|---:|---|
| `financial_negative_tone` | 0..1 | Core LM/Tetlock-style signal; useful, but noisy alone. |
| `financial_positive_tone` | 0..1 | Needed to compute tone balance and management optimism. |
| `uncertainty_modal_intensity` | 0..1 | Uncertainty and modal language repeatedly matter in filing text. |
| `forward_looking_guidance_direction` | -1..1 | Captures future-looking MD&A/guidance direction. |
| `risk_factor_specificity` | 0..1 | Separates actionable Item 1A risk from generic boilerplate. |
| `litigation_regulatory_risk` | 0..1 | LM litigious category and risk-factor literature support it. |
| `financial_constraint_liquidity_stress` | 0..1 | Supported by 10-K financial-constraint text literature. |
| `earnings_release_manager_tone` | -1..1 | Earnings-release tone can signal expected future performance. |
| `readability_obfuscation_risk` | 0..1 | Guards against hard-to-read disclosure and impression management. |
| `numeric_fundamental_evidence_density` | 0..1 | Distinguishes real operating evidence from pure wording. |
| `macro_downside_pressure` | 0..1 | Connects official macro/rates/credit/inflation docs to portfolio risk. |
| `text_signal_confidence` | 0..1 | Prevents PPO from treating weak boilerplate as strong signal. |

## What To Avoid

- Generic sentiment as the only feature.
- Raw document length as an alpha signal without source controls.
- Risk-factor word counts that do not distinguish new/specific risk from boilerplate.
- Features that are nearly always zero after daily panel aggregation.
- LLM labels trained or evaluated using future returns without strict date controls.

## Source-Specific Extraction Guidance

| Source family | Strong features | Weak / risky features |
|---|---|---|
| SEC risk factors / 10-K / 10-Q | `risk_factor_specificity`, `litigation_regulatory_risk`, `uncertainty_modal_intensity`, `readability_obfuscation_risk` | Raw positive tone is usually less useful here. |
| MD&A / financial report | `forward_looking_guidance_direction`, `financial_constraint_liquidity_stress`, `readability_obfuscation_risk` | Generic future-looking density without direction. |
| Earnings releases / 8-K exhibits | `earnings_release_manager_tone`, `numeric_fundamental_evidence_density`, `forward_looking_guidance_direction` | Promotional tone without numeric support. |
| Company IR / press releases | `numeric_fundamental_evidence_density`, `financial_positive_tone`, `text_signal_confidence` | Product-launch PR tone can be marketing noise. |
| Official macro | `macro_downside_pressure`, `uncertainty_modal_intensity`, `text_signal_confidence` | Company-specific event risk. |

## References

- Loughran and McDonald, "When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks", Journal of Finance.
- Tetlock, "Giving Content to Investor Sentiment: The Role of Media in the Stock Market", Journal of Finance.
- Li, "Annual report readability, current earnings, and earnings persistence", Journal of Accounting and Economics.
- Campbell, Chen, Dhaliwal, Lu, and Steele, "The information content of mandatory risk factor disclosures in corporate filings", Review of Accounting Studies.
- Li, "The Information Content of Forward-Looking Statements in Corporate Filings", Journal of Accounting Research.
- Davis, Piger, and Sedor, "Beyond the Numbers: Measuring the Information Content of Earnings Press Release Language".
- Bodnaruk, Loughran, and McDonald, "Using 10-K Text to Gauge Financial Constraints", Journal of Financial and Quantitative Analysis.
- Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models".
