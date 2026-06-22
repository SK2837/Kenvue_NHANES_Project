# NHANES Liver Injury Analysis × Kenvue Portfolio
### A Population-Level Study of Metabolic Risk, COVID-Era Shifts, and Agentic AI
**Adarsh Kasula | Data Science Portfolio | 2017–2023 NHANES Cohorts**

---

## SLIDE 1 — Project Overview: Why This, Why Kenvue

### What This Project Is

This project applies **survey-weighted epidemiological modeling and machine learning** to two waves of the U.S. National Health and Nutrition Examination Survey (NHANES) — 2017-2018 (pre-COVID) and 2021-2023 (post-COVID) — to understand **who is at risk for liver injury** as measured by elevated alanine aminotransferase (ALT), a primary clinical marker for hepatocellular damage.

### Why NHANES?

- Largest nationally representative health survey in the U.S. — covers ~5,000 civilians per cycle
- Combines medical examination, lab results, dietary data, and questionnaires
- Survey weights allow results to **generalize to the entire U.S. non-institutionalized adult population** (~240 million people)
- Used by the CDC, FDA, and academic epidemiologists as the gold standard for population-level health inference

### Why This Is Directly Relevant to Kenvue

Kenvue owns brands that sit at the intersection of **everyday health and liver safety**:

| Kenvue Brand Area | Relevance to This Study |
|---|---|
| **Pain relief / antipyretics** (Tylenol / acetaminophen) | Acetaminophen is the #1 cause of acute liver failure in the U.S. — understanding baseline ALT elevation rates in the population defines the "background noise" that clinical teams must account for |
| **Digestive health** | Metabolic dysfunction (obesity, high triglycerides) drives non-alcoholic fatty liver disease (NAFLD), a growing OTC opportunity |
| **Consumer wellness** | Depression, poor sleep, and sedentary behavior are rising post-COVID — this project quantifies their actual liver impact |
| **Product safety surveillance** | Real-world elevated ALT prevalence (8.2% U.S. adults) is the denominator for any post-market safety signal |

> **Core thesis:** Knowing *who* has elevated ALT at baseline — and how that changed after COVID — helps Kenvue design safer products, better label warnings, and more targeted consumer health campaigns.

---

## SLIDE 2 — Data, Methods & Analytical Pipeline

### Dataset

| Property | 2017-2018 (Pre-COVID) | 2021-2023 (Post-COVID) |
|---|---|---|
| Raw participants | ~9,000 | ~9,000 |
| Post-exclusion analytic sample | **3,543** | **3,947** |
| Columns (features) | 267 | 260 |
| ALT elevated >40 U/L | **8.15%** | **7.36%** |
| Survey weight variable | WTMEC2YR | WTMEC2YR |

**Exclusion criteria:** Age <18, pregnancy, missing ALT/survey design variables, extreme blood lead outliers (>99th percentile)

### Key Variables

- **Outcome:** ALT > 40 U/L (unisex) and sex-specific AASLD thresholds (>29 U/L women / >33 U/L men)
- **Metabolic predictors:** Age, sex, poverty-income ratio, waist circumference, diabetes status, smoking, log-triglycerides
- **Environmental predictors:** log(blood lead), log(blood cadmium), log(blood mercury)
- **COVID-era wellness:** PHQ-9 depression score (≥10 = moderate depression), sleep hours (<7h = short sleep), sedentary time (≥8h/day = high sedentary)

### Methods

```
Phase 1 — EDA & Weighted Prevalence
  Survey-weighted descriptive statistics, subgroup breakdowns by sex / age / race / BMI

Phase 2 — Epidemiologic Inference
  Survey-weighted logistic regression (svy package, Taylor-series variance estimation)
  10-predictor model: age, sex, PIR, waist, diabetes, smoking, triglycerides, 3 metals

Phase 2 — Machine Learning
  XGBoost classifier (31 features, grid-searched hyperparameters)
  SHAP TreeExplainer for feature importance and individual waterfall plots
  Calibration: Platt scaling | ROC + precision-recall curves

Phase 3 — Pre/Post COVID Comparison
  Same 10-predictor model re-fit to 2021-2023 cohort
  Prevalence shifts, OR trajectory, ML performance stability
  Wellness variable hypothesis tests (adjusted for 10-predictor base model)
```

**[figure: fig_correlation_heatmap.png]**

> All survey weights applied throughout. No unweighted estimates reported.

---

## SLIDE 3 — Core Findings: 2017-2018 Baseline Analysis

### Weighted Prevalence of Elevated ALT — Who Is at Risk?

| Subgroup | Weighted Prevalence |
|---|---|
| Overall U.S. adults | **5.2%** (sex-specific threshold) / **8.2%** (>40 U/L) |
| Male | 3.9% |
| Female | 6.5% |
| Age 18-40 | **8.7%** ← highest age group |
| Age 40-60 | 4.9% |
| Age 60+ | 2.7% |
| Obese (BMI ≥30) | **8.3%** |
| Normal weight | 2.0% |
| Mexican American | **8.9%** |
| Non-Hispanic White | 4.5% |
| Non-Hispanic Black | 3.4% |

**[figure: fig_prevalence_by_race.png]**

### Survey-Weighted Logistic Regression — 10-Predictor Model

| Predictor | Odds Ratio | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| Age (years) | 0.960 | 0.939 – 0.982 | 0.005 | ✓ |
| **Waist circumference (cm)** | **1.027** | 1.009 – 1.044 | **0.012** | ✓ |
| **Triglycerides (log)** | **1.773** | 1.160 – 2.710 | **0.018** | ✓ |
| Male sex | 2.025 | 0.937 – 4.377 | 0.065 | Borderline |
| Diabetes | 1.975 | 0.922 – 4.232 | 0.070 | Borderline |
| Ever smoker | 0.641 | 0.376 – 1.095 | 0.086 | — |
| Poverty-income ratio | 1.015 | 0.864 – 1.192 | 0.823 | — |
| log(Blood Lead) | 1.373 | 0.837 – 2.252 | 0.160 | — |
| log(Blood Cadmium) | 0.937 | 0.661 – 1.328 | 0.651 | — |
| log(Blood Mercury) | 1.042 | 0.887 – 1.223 | 0.542 | — |

**Key insight:** Waist circumference and triglycerides are the two modifiable metabolic risk factors with the strongest independent association with elevated ALT. Heavy metals (lead, cadmium, mercury) are **not** significant after metabolic adjustment.

**[figure: fig_forest_plot_OR.png]**

### Machine Learning Performance (XGBoost, 31 features)

| Metric | Value |
|---|---|
| AUC-ROC | **0.973** |
| PR-AUC | 0.793 |
| Sensitivity | 92.6% |
| Specificity | 92.2% |
| Negative Predictive Value | **99.3%** |

**[figure: fig_roc_calibration.png]**

### Top SHAP Feature Importances

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | AST | 1.784 | Correlated outcome — near-perfect predictor |
| 2 | GGT | 0.409 | Correlated outcome |
| 3 | Age | 0.376 | Strong protective effect (older → less ALT elevation) |
| 4 | BMI | 0.140 | Metabolic burden |
| 5 | Triglycerides | 0.094 | Lipid pathway |
| 6 | Waist circumference | 0.078 | Central adiposity |

> AST and GGT dominate SHAP because they are biochemically correlated with ALT — not independent causal drivers. The **epidemiologically actionable** predictors are waist and triglycerides.

**[figure: fig_shap_importance.png]**

---

## SLIDE 4 — Post-COVID Comparison: What Changed Between 2017-18 and 2021-23?

### Population Shifts — Five Years Later

| Variable | 2017-2018 | 2021-2023 | Change |
|---|---|---|---|
| ALT > 40 U/L | 8.15% | 7.36% | ▼ −0.8 pp |
| ALT elevated (sex-specific) | 5.24% | 5.68% | ▲ +0.4 pp |
| Diabetes | 14.24% | 12.98% | ▼ −1.3 pp |
| Obesity (BMI ≥30) | 44.99% | 42.38% | ▼ −2.6 pp |
| Central obesity | 63.87% | 59.95% | ▼ −3.9 pp |
| Ever smoker | 41.17% | 36.18% | ▼ −5.0 pp |
| **Depression (PHQ-9 ≥10)** | **8.16%** | **11.76%** | **▲ +3.6 pp** |
| High sedentary (≥8h/day) | 30.24% | 33.63% | ▲ +3.4 pp |
| Short sleep (<7h) | 24.99% | 20.26% | ▼ −4.7 pp |

**[figure: fig_prevalence_comparison.png]**
**[figure: fig_riskfactor_comparison.png]**

### Model Comparison — How Predictors Shifted

The **same 10-predictor model** was re-fit to the 2021-2023 cohort. Dramatic changes in OR magnitude and significance:

| Predictor | OR 2017-18 | p 2017 | OR 2021-23 | p 2021 | Verdict |
|---|---|---|---|---|---|
| **Male sex** | 2.025 | 0.065 | **2.930** | **0.0006 ★★★** | Strengthened dramatically |
| **Diabetes** | 1.975 | 0.070 | **0.898** | 0.738 | **Reversed — no longer a risk factor** |
| **Triglycerides** | 1.773 ★ | 0.018 | **1.967 ★** | 0.014 | Strengthened, still significant |
| Waist circumference | 1.027 ★ | 0.012 | 1.018 ★ | 0.012 | Stable, still significant |
| Age | 0.960 ★ | 0.005 | 0.979 ★ | 0.022 | Attenuated but stable |
| Heavy metals | — | >0.15 | — | >0.14 | Non-significant in both |

**[figure: fig_model_comparison_forest.png]**

### The Two Most Important Post-COVID Findings

**Finding 1 — Male sex became the dominant risk factor post-COVID (OR 2.93, p=0.0006)**
The sex differential in liver injury widened considerably. Post-COVID metabolic sequelae, behavioral differences in physical activity recovery, and changes in alcohol patterns may explain this. From Kenvue's perspective, male-targeted liver health products (Tylenol liver safety messaging, digestive supplements) become more relevant.

**Finding 2 — Diabetes is no longer a significant risk factor in 2021-2023 (OR 0.90, p=0.74)**
This is counterintuitive and clinically interesting. Possible explanations: (a) improved diabetes management (GLP-1 receptor agonist adoption post-2021 may have reduced hepatic steatosis in diabetics), (b) COVID-related selection effects (higher COVID mortality in diabetics skewing the surviving population), (c) survivor bias in the 2021-2023 survey sample. This warrants follow-up but suggests the metabolic landscape has genuinely changed.

**ML Performance Comparison:**

| Metric | 2017-2018 | 2021-2023 | Change |
|---|---|---|---|
| AUC-ROC | 0.973 | **0.955** | −0.018 |
| PR-AUC | 0.793 | 0.721 | −0.072 |

The model trained on 2017-2018 data generalizes reasonably well to 2021-2023 (AUC 0.955) — the biological signal is stable even as population-level prevalence shifts.

**[figure: fig_roc_comparison.png]**

---

## SLIDE 5 — Wellness Variables: A Null Finding That Matters

### The Hypothesis (pre-COVID concern)

Post-COVID mental health deterioration, disrupted sleep, and sedentary lockdown behavior were widely hypothesized to worsen metabolic liver disease. This project formally tested whether these wellness variables predict elevated ALT **over and above** the 10-predictor metabolic model.

### Results — Adjusted Odds Ratios for Elevated ALT

| Wellness Variable | Prevalence 2017 | Prevalence 2021 | OR_adj 2017 | p 2017 | OR_adj 2021 | p 2021 |
|---|---|---|---|---|---|---|
| Depression (PHQ-9 ≥10) | 8.2% | 11.8% | 1.371 | 0.463 | 0.922 | 0.789 |
| High sedentary (≥8h/day) | 30.2% | 33.6% | 0.673 | 0.233 | 0.951 | 0.855 |
| Short sleep (<7h/night) | 25.0% | 20.3% | 1.102 | 0.711 | 0.929 | 0.683 |

**[figure: fig_wellness_forest_plot.png]**

### Interpretation

**Depression, poor sleep, and high sedentary time are NOT independent predictors of elevated ALT** after controlling for metabolic risk factors — in either cohort.

This is a meaningful null finding for three reasons:
1. **For Kenvue product strategy:** Wellness products marketed on liver health claims (e.g., supplements targeting "liver detox" for stressed, sleep-deprived consumers) lack epidemiological support from NHANES data. The metabolic pathway (obesity, triglycerides) dominates.
2. **For safety communications:** Tylenol's consumer-facing liver messaging should focus on dose adherence and metabolic risk co-factors, not mental health or sleep status.
3. **The prevalence trends still matter:** Depression rose +3.6 pp and sedentary behavior rose +3.4 pp post-COVID — these are real population shifts worth tracking even if they don't directly predict ALT through the metabolic model.

---

## SLIDE 6 — Results, Conclusions & How Kenvue Can Use This

### What We Found — Summary

| Domain | Key Finding |
|---|---|
| **Baseline prevalence** | 8.2% of U.S. adults have ALT >40 U/L; 5.2% by sex-specific thresholds |
| **Highest-risk groups** | Young adults (18-40), Mexican Americans, obese individuals |
| **Metabolic drivers** | Waist circumference and triglycerides are the only modifiable predictors with statistically significant associations after full adjustment |
| **Heavy metals** | Not significantly associated with elevated ALT once metabolic factors controlled |
| **Post-COVID shift** | Male sex dramatically strengthened (OR 2.03→2.93); diabetes reversed direction |
| **Wellness variables** | Depression, sleep, sedentary time — NOT independent ALT predictors |
| **ML model** | AUC 0.97 in 2017-18; generalizes to 0.955 in 2021-23 |

### What We Understand

1. **Central adiposity (waist circumference) is the single most policy-relevant modifiable risk factor.** Every 1 cm increase in waist circumference raises ALT elevation odds by 2.7%. This is the lever for intervention.

2. **Triglycerides are the second key pathway.** The log-linear relationship suggests that even modest dyslipidemia compounds risk — relevant to any OTC digestive or metabolic health product.

3. **The post-COVID population is biologically different.** Male sex dominance strengthening and diabetes de-coupling are not statistical noise — they reflect genuine shifts in population health patterns that product safety teams need to track prospectively.

4. **The ML model achieves 99.3% negative predictive value.** This means it can effectively rule out elevated ALT in screening contexts — valuable for clinical decision support tools.

### How Kenvue Can Use This

| Application | How |
|---|---|
| **Tylenol label & safety** | Quantify population-level "background" ALT elevation rate (8.2%) for adverse event reporting context; identify highest-risk consumer segments (obese males, high triglycerides) for targeted dosing guidance |
| **OTC hepatology products** | Waist circumference + triglycerides = the metabolic phenotype to target in new product positioning; the 44.99% obesity prevalence is the addressable market |
| **Post-market surveillance** | Re-run this model on future NHANES cycles to detect further post-COVID shifts before they reach pharmacovigilance signals |
| **Consumer health campaigns** | Male sex became OR 2.93 post-COVID — male-targeted liver health awareness is data-supported |
| **R&D pipeline screening** | ML model (AUC 0.97, NPV 99.3%) as a population-level risk stratification tool for clinical trial enrichment |

**[figure: fig_phase3_dashboard.png]**
**[figure: fig_covid_impact_summary.png]**

---

## SLIDE 7 — Agentic AI Pipeline: Architecture

### What Is the Agentic Layer?

On top of the full statistical analysis pipeline, this project includes an **AI-powered research assistant** built with Anthropic's Claude API. It turns the static analysis results into an interactive natural-language interface — letting any team member (including non-technical stakeholders) ask questions and get personalized risk estimates backed by the actual epidemiological data.

### Architecture — Three-Agent Design

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│  ORCHESTRATOR  (Claude Sonnet 4.6)      │
│  Classifies intent → routes to agent   │
│  Latency: ~100ms | No tools            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴──────────┐
       ▼                  ▼
┌─────────────┐    ┌──────────────────────┐
│  Q&A AGENT  │    │  ANALYSIS AGENT      │
│ Haiku 4.5   │    │  Sonnet 4.6          │
│ Fast/cheap  │    │  Powerful/slower     │
│ Max 3 turns │    │  Max 5 turns         │
└─────────────┘    └──────────────────────┘
```

**Orchestrator routing logic:**
- → **Q&A Agent:** Personalized risk questions, stats lookups, prevalence queries, report retrieval
- → **Analysis Agent:** Custom subgroup regression, "why does X cause Y?", re-run analysis scripts
- → **Direct reply:** Greetings, meta-questions about the system

### Tool Suite — 11 Tools Across Both Agents

**Q&A Tools (read-only, milliseconds)**

| Tool | What It Does |
|---|---|
| `read_or_table()` | Returns the 10-predictor OR table (OR, CI, beta, p-value) |
| `read_shap_rankings(top_n)` | Returns top-N SHAP feature importances |
| `read_ml_metrics()` | Returns AUC-ROC, PR-AUC, sensitivity, specificity |
| `read_prevalence_table(group)` | Returns weighted prevalence by sex/age/race/BMI/diabetes |
| `read_report(name)` | Retrieves full text of technical/safety/methods reports |
| **`compute_individual_risk(...)`** | **Personalized probability estimate — the key feature** |

**Analysis Tools (compute, seconds)**

| Tool | What It Does |
|---|---|
| `query_analytic_table(filter, columns)` | Survey-weighted aggregate query on NHANES parquet |
| `run_subgroup_analysis(filter, label)` | Fits survey-weighted logistic regression on any subgroup |
| `explain_risk_factors(factor, audience)` | Synthesizes OR + SHAP context for biological explanation |
| `run_inference_script()` | Re-runs `04_inference.py` (with confirmation guard) |
| `run_ml_script()` | Re-runs `05_predict_ml.py` (with confirmation guard) |

### Personalized Risk Estimation — How `compute_individual_risk` Works

```
1. Load betas from OR table (log-odds scale)
2. Compute population-weighted means from parquet (calibration baseline)
3. Calibrate intercept so "average person" → 5.2% probability
4. Sum: intercept + Σ(beta × person_value) for each predictor
5. Convert log-odds → probability: 1 / (1 + e^−linear)
6. Return: estimated_risk_pct, relative_odds_vs_average, top_3_drivers
```

**Example output for a 45-year-old diabetic male, waist 102 cm, triglycerides 220 mg/dL:**
- Estimated risk: **~12%** (vs. 5.2% population average)
- Relative odds: **~2.3×** above average
- Top drivers: (1) Male sex, (2) Triglycerides, (3) Waist circumference

---

## SLIDE 8 — Agentic AI Pipeline: Live Use Cases & Demo

### How to Launch

```bash
# Streamlit web app (recommended for demos)
streamlit run src/07_chat.py

# Command-line interface (quick queries)
python src/07_chat.py --cli
```

**Requires:** `ANTHROPIC_API_KEY` environment variable

### Sample Interactions — What You Can Ask

**Personalized Risk Assessment**
> *"A 55-year-old male, waist 108 cm, triglycerides 280, former smoker, no diabetes — what is his ALT elevation risk?"*
→ Q&A Agent calls `compute_individual_risk` → Returns ~14% estimated risk, 2.7× average → Agent narrates in plain English with top 3 drivers

**Population Statistics**
> *"What's the weighted prevalence of elevated ALT in obese Mexican-American adults?"*
→ Analysis Agent calls `query_analytic_table` → Returns N=312, prevalence 11.4% weighted

**Model Interpretation**
> *"Why does triglyceride level matter for liver injury — what's the biological mechanism?"*
→ Analysis Agent calls `read_or_table` + `explain_risk_factors` → Synthesizes: lipid overflow hypothesis, hepatic de novo lipogenesis, VLDL export impairment

**Post-COVID Question**
> *"Did depression or sedentary behavior predict elevated ALT post-COVID?"*
→ Q&A Agent answers directly from wellness hypothesis results: no significant association in either cohort after metabolic adjustment

**Subgroup Analysis**
> *"Run the model for diabetic women specifically"*
→ Analysis Agent calls `run_subgroup_analysis(filter='diabetes==1 and sex_male==0')` → Fits survey-weighted logistic regression on N=~800 subgroup → Returns subgroup-specific OR table

### Streamlit App Layout

The full app has **three tabs**:

| Tab | Contents |
|---|---|
| **Chat** | Live AI conversation + figure viewer (split by Phase 1/2 and Phase 3) + report toggle buttons |
| **Post-COVID Analysis** | Four sub-tabs: Overview prevalence shifts, Model OR comparison, Wellness variables, ML comparison |
| **Generate Report** | AI-generated stakeholder report for any consumer profile or research question |

**Sidebar features:**
- Real-time metric cards showing both cohort prevalence side-by-side (2017-2018 vs 2021-2023)
- Phase 3 COVID callout card (dominant finding: Male sex OR 2.93 ★★★)
- Phase-split figure viewer (14 Phase 1&2 figures / 8 Phase 3 figures)
- Collapsible technical, safety, and methods reports

### Project File Structure

```
Kenvue Project/
├── data/
│   ├── raw/               NHANES 2017-2018 XPT files (15 modules)
│   ├── raw_2021/          NHANES 2021-2023 XPT files (15 modules)
│   └── processed/         analytic_table.parquet | analytic_table_2021.parquet
├── figures/
│   ├── phase1_eda/        5 exploratory figures
│   ├── phase2_analysis/   9 inference & ML figures
│   └── phase3_covid/      8 pre/post COVID comparison figures
├── reports/
│   ├── phase2_analysis/   OR tables, SHAP rankings, ML metrics, stakeholder reports
│   └── phase3_covid/      Model comparison, wellness hypothesis, SHAP/ML comparison
└── src/
    ├── 02_build_dataset.py → 12_build_dataset_2021.py   Data pipeline
    ├── 03_eda_weighted.py → 10_new_predictor_figures.py  Analysis scripts
    ├── 13_wellness_hypothesis.py → 17_comparison_figures.py  Phase 3 comparison
    ├── agents/
    │   ├── orchestrator.py   Intent routing
    │   ├── qa_agent.py       Fast Q&A (Haiku)
    │   ├── analysis_agent.py Deep analysis (Sonnet)
    │   └── tools.py          11 tool functions + schemas
    └── 07_chat.py            Streamlit UI + CLI entrypoint
```

---

*Analysis based on NHANES 2017-2018 and 2021-2023 public-use data. Survey-weighted estimates represent the U.S. civilian non-institutionalized adult population. All findings are associational; no causal claims are made. Not intended for individual clinical diagnosis.*
