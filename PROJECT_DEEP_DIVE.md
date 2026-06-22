# NHANES Exposome-Wide Liver-Injury Screen — Deep Technical Reference

> This document is the intellectual map of the project. It explains every design decision, every correctness rule, and how all components connect. It is not a tutorial — it assumes working knowledge of epidemiology, regression, and Python data science. Cross-references to `ROADMAP.md` are noted where relevant.

---

## Table of Contents

1. [Problem & Motivation](#1-problem--motivation)
2. [Data Sources](#2-data-sources)
3. [Outcome Definition](#3-outcome-definition)
4. [Exclusion Criteria](#4-exclusion-criteria)
5. [Missing Data Handling](#5-missing-data-handling)
6. [Survey Design](#6-survey-design)
7. [Weighted EDA](#7-weighted-eda)
8. [Inference Model: Survey-Weighted Logistic Regression](#8-inference-model-survey-weighted-logistic-regression)
9. [ML Prediction Model: XGBoost + SHAP](#9-ml-prediction-model-xgboost--shap)
10. [LLM-Assisted Reporting Layer](#10-llm-assisted-reporting-layer)
11. [Correctness Rules](#11-correctness-rules)
12. [Deliverables Map](#12-deliverables-map)
13. [What's New vs. The Field](#13-whats-new-vs-the-field)
14. [Limitations & Future Work](#14-limitations--future-work)
15. [AI Usage Appendix](#15-ai-usage-appendix)

---

## 1. Problem & Motivation

### Why elevated ALT?

Alanine aminotransferase (ALT) is the most sensitive and specific serum marker of hepatocellular injury in routine clinical chemistry panels. The liver is the primary site of cytochrome P450-mediated xenobiotic metabolism, meaning that any compound processed hepatically — including virtually all orally-administered consumer-health products — perturbs the same biochemical machinery measured by ALT. Population-level characterization of what drives baseline ALT elevation is therefore not merely epidemiological curiosity: it defines the background noise against which any ingredient's hepatic safety signal must be detected.

### Why NHANES 2017–2018?

The National Health and Nutrition Examination Survey (NHANES) is the only nationally representative survey of the U.S. civilian non-institutionalized population that combines:
- Direct physical examination (phlebotomy, anthropometrics)
- Heavy-metal biomarker assays (blood lead, cadmium, total mercury)
- Full clinical chemistry panels (ALT, AST, ALP, total bilirubin, albumin, glucose, lipids)
- Hepatitis serology (enabling exclusion of viral hepatitis)
- Detailed behavioral data (alcohol, smoking, diet recall)
- A complex probability sample design with published design variables

The 2017–2018 cycle (suffix `_J`) is the most recent pre-pandemic cycle with a complete environmental-chemical subsample, making it the best available dataset for an exposome-wide screen. Post-2019 cycles were disrupted by COVID-19 field suspension.

### What the prior literature does wrong

A body of 2021–2025 NHANES machine-learning papers predicts liver outcomes (ALT elevation, NAFLD, MASLD) using XGBoost, random forest, and neural networks. The near-universal error is treating the NHANES complex sample as a simple random sample:

- They compute prevalence as row proportions, not weighted proportions → biased estimates
- They run regression without survey design correction → artificially narrow confidence intervals (false precision) because the effective sample size is smaller than N
- They report model AUC without noting the model is trained on a convenience sample, not a probability sample → predictions cannot be extrapolated to the U.S. population

This project fixes both problems: survey-correct inference for population-level claims, clearly labeled ML for individual-level prediction.

### Why this matters for safety science

The same computational pipeline that identifies which population exposures drive baseline liver risk is the pipeline you would apply to a candidate ingredient's biomarker signature in a real-world evidence study. The transferable contribution is the method — survey-weighted feature screening — not just the specific finding about ALT.

---

## 2. Data Sources

All files are accessed from the CDC NHANES 2017–2018 data release. Base URL pattern:
```
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{MODULE_CODE}.XPT
```

### Module inventory

| Module Code | Full Name | Key Variables Extracted | Role in Analysis |
|-------------|-----------|------------------------|-----------------|
| DEMO_J | Demographic Variables | SEQN, RIDAGEYR, RIAGENDR, RIDRETH3, INDFMPIR, WTMEC2YR, SDMVSTRA, SDMVPSU | Base frame; survey design variables; demographics |
| BMX_J | Body Measures | BMXBMI, BMXWAIST | Anthropometric predictors |
| BIOPRO_J | Standard Biochemistry Panel | LBXSATSI (ALT — outcome), LBXSASSI (AST), LBXSAPSI (ALP), LBXSTB (total bilirubin), LBXSAL (albumin), full biochemistry panel | Primary outcome + biochemical predictor panel |
| PBCD_J | Cadmium, Lead, Total Mercury, Selenium, Manganese | LBXBPB (blood lead), LBXBCD (blood cadmium), LBXTHG (blood mercury) | Key environmental exposures |
| ALQ_J | Alcohol Use | ALQ010 (drink ever), ALQ120Q (drinks/week), ALQ510 | Behavioral predictor; exclusion criterion |
| DIQ_J | Diabetes | DIQ010 (doctor-diagnosed diabetes) | Comorbidity predictor |
| SMQ_J | Smoking — Cigarette Use | SMQ020 (ever smoked), SMQ040 (current smoker) | Behavioral predictor; cadmium confounder |
| HEPB_S_J | Hepatitis B Surface Antibody & Antigen | LBXHBS (surface antigen) | Exclusion criterion |
| HEPC_J | Hepatitis C Antibody | LBXHCR (HCV antibody) | Exclusion criterion |

### Merge strategy

The merge is always a **left join onto DEMO_J using SEQN as the sole key**.

Rationale: DEMO_J contains all sampled participants, including those who completed only the household interview and not the mobile examination center (MEC) visit. Left-joining preserves the full sample structure needed to correctly construct the survey design object. Participants who did not attend the MEC will have `WTMEC2YR = 0` and will be excluded later — but they must remain in the frame until after the design object is constructed.

**Never inner-join before building the survey design object.** Inner-joining drops non-examined participants silently and corrupts the design weights.

### Variable labeling

NHANES variable names are cryptic (e.g., `LBXSATSI`). A codebook CSV (`/data/processed/codebook.csv`) maps each variable to:
- Module of origin
- Plain-English label
- Unit of measure
- Expected range

LLM assistance (Claude) was used to batch-translate variable codes to plain-English labels from the NHANES web codebook. All labels were verified against the official CDC documentation.

---

## 3. Outcome Definition

### Primary outcome: sex-specific ALT threshold

Binary outcome `ALT_elevated`:

| Sex | Threshold | Source |
|-----|-----------|--------|
| Male (RIAGENDR = 1) | LBXSATSI > 56 U/L | AASLD Practice Guidance (Kwo et al., 2017) |
| Female (RIAGENDR = 2) | LBXSATSI > 33 U/L | AASLD Practice Guidance (Kwo et al., 2017) |

Sex-specific thresholds are used because ALT reference ranges differ by sex due to differences in lean body mass, hormonal milieu, and body composition. Using a single unisex threshold (often 40 U/L) misclassifies a substantial fraction of women as normal when they have biochemically significant liver injury.

**Coding logic:**
```python
conditions = [
    (df["RIAGENDR"] == 1) & (df["LBXSATSI"] > 56),   # Male, elevated
    (df["RIAGENDR"] == 2) & (df["LBXSATSI"] > 33),   # Female, elevated
]
choices = [1, 1]
df["ALT_elevated"] = np.select(conditions, choices, default=0)
df.loc[df["LBXSATSI"].isna(), "ALT_elevated"] = np.nan
```

### Sensitivity analysis outcome

`ALT_elevated_40`: both sexes, threshold = 40 U/L. Used to test whether key findings are threshold-dependent. If the direction and magnitude of major predictors are stable across both thresholds, findings are robust. If they change, the threshold choice must be explicitly discussed.

### Expected prevalence

Using AASLD sex-specific thresholds in NHANES populations, weighted prevalence of elevated ALT is typically 7–11% in non-hepatitis, non-excessive-alcohol adults. Values outside this range indicate a data or coding error.

---

## 4. Exclusion Criteria

Exclusions are applied sequentially and documented in a flow table. Each exclusion has an explicit scientific justification — not just a convention.

### Criterion 1: Adults 18+ (RIDAGEYR ≥ 18)

**Why:** Pediatric ALT reference ranges and the etiology of liver injury differ substantially from adults. Mixing pediatric and adult participants would require age-interaction terms in every model and would complicate the public-health interpretation. The consumer-health safety context is adult-focused.

### Criterion 2: MEC-examined with positive weight (WTMEC2YR > 0)

**Why:** Only MEC-examined participants have biomarker data. Participants with `WTMEC2YR = 0` completed only the household interview and contribute no outcome or exposure data. However, they must remain in the frame until the design object is constructed — then this criterion is applied as an analytic filter.

### Criterion 3: Non-missing ALT outcome (LBXSATSI not NaN)

**Why:** Cannot classify an individual without the outcome measurement. Participants missing ALT are excluded from the analytic sample but their weights are not redistributed — a limitation acknowledged in the methods.

### Criterion 4: Hepatitis B surface antigen negative (LBXHBS ≠ 1)

**Why:** Chronic hepatitis B directly causes hepatocellular inflammation and elevated ALT through viral replication, not xenobiotic metabolism. Including HBsAg-positive individuals would confound the relationship between environmental exposures and ALT because the dominant causal pathway for their liver injury is viral, not chemical. These participants represent ~0.3–0.5% of U.S. adults.

### Criterion 5: Hepatitis C antibody negative (LBXHCR ≠ 1)

**Why:** Same rationale as Hepatitis B. HCV RNA-positive individuals have liver injury driven by viral pathophysiology, not chemical exposure. HCV prevalence in NHANES is ~1–2% in adults.

### Criterion 6: Non-excessive alcohol use

**Why:** Alcoholic liver disease is a distinct pathophysiology (hepatic steatosis → alcoholic hepatitis → cirrhosis) driven by acetaldehyde toxicity, not xenobiotic CYP metabolism in the usual sense. Including heavy drinkers would confound the heavy-metal and biochemical exposure associations.

**Operationalization:**
- Male: exclude if `ALQ510 > 14` (more than 14 drinks/week on average)
- Female: exclude if `ALQ510 > 7` (more than 7 drinks/week)
- This matches the NIAAA definition of "heavy drinking" and mirrors the approach in leading NHANES liver studies (e.g., Younossi et al.)
- Participants who report never drinking are retained

**Edge case:** `ALQ510` may be NaN for never-drinkers (they skip the alcohol quantity questions). These participants are retained (coded as 0 drinks/week effectively).

---

## 5. Missing Data Handling

### The NHANES sentinel code problem

NHANES uses numeric codes — not missing values (`NaN`) — to indicate non-response in interview-administered questionnaires. If these are not recoded before analysis, they become valid numeric observations in statistical models, producing catastrophically wrong results (e.g., a participant who "refused" a question coded 7 becomes indistinguishable from someone who answered 7 on a 1–10 scale).

### Sentinel code table

| Code | Meaning | Context |
|------|---------|---------|
| 7 | Refused | Single-digit categorical items |
| 77 | Refused | Two-digit numeric items |
| 777 | Refused | Three-digit numeric items |
| 9 | Don't know | Single-digit categorical items |
| 99 | Don't know | Two-digit numeric items |
| 999 | Don't know | Three-digit numeric items |

### Recoding approach

Applied variable-by-variable, not blanket-across-all-columns (blanket replacement would corrupt continuous variables that legitimately have values of 7, 9, 77, etc.).

The codebook CSV documents which variables receive which sentinel recoding. Variables needing explicit attention:
- `ALQ510` (drinks/week): codes 777 = refused, 999 = don't know
- `DIQ010` (diabetes): code 7 = refused, 9 = don't know
- `SMQ020` (ever smoked): code 7 = refused, 9 = don't know
- `LBXHBS`, `LBXHCR`: code 3 = indeterminate (treat as NaN for exclusion)

Laboratory measurements (BIOPRO_J, PBCD_J) are continuous and use `.` for missing in the XPT file, which `pyreadstat` correctly loads as `NaN`. No sentinel recoding needed for lab values.

### Why NOT do listwise deletion before survey design

Listwise deletion (dropping all rows with any NaN) before constructing the survey design object changes the sample structure in a way that is not correctable by the design weights. The design was calibrated to the full examined sample. Dropping rows removes the variance structure that the strata/PSU variables encode.

**Correct approach:**
1. Construct survey design object on the full MEC-examined sample (with NaN present)
2. Let the modeling function handle missingness within its own missing-data protocol (most will use complete cases per-model, which is acceptable when the outcome and key predictors are not systematically missing)
3. Document the number of complete cases for each model separately

### Multiple imputation (not implemented in core; stretch)

For a rigorous analysis, multiple imputation would be the preferred approach. The core project uses complete-case analysis with documentation of missingness rates. A `stretch_imputation.py` can be added if time permits, using `sklearn.impute.IterativeImputer` or `miceforest`.

---

## 6. Survey Design

### What "complex sample design" means

NHANES is not a simple random sample of U.S. adults. It uses a four-stage stratified probability sample:
1. **Primary sampling units (PSUs):** counties or groups of contiguous counties, stratified by geography and socioeconomic variables
2. **Secondary sampling units:** census tracts within PSUs
3. **Tertiary sampling units:** household clusters within tracts
4. **Individuals:** persons within households, with oversampling of certain subgroups (elderly, low-income, racial/ethnic minorities)

Because of this design, a single participant may statistically "represent" 10,000–50,000 U.S. adults (their weight `WTMEC2YR`). Ignoring this in analysis produces two errors:
1. **Biased estimates:** prevalence estimates are not representative of the U.S. population
2. **False precision:** standard errors are underestimated because sampled individuals within the same PSU are more similar to each other than a random draw would produce (intraclass correlation)

### The three design variables

| Variable | Role |
|----------|------|
| `WTMEC2YR` | Probability weight (sum across analytic sample ≈ U.S. adult civilian population) |
| `SDMVSTRA` | Masked stratum identifier (used for variance estimation via Taylor-series linearization) |
| `SDMVPSU` | Masked PSU identifier (2 PSUs per stratum in NHANES masked design) |

### How these enter the analysis

In `samplics`:
```python
from samplics.estimation import TaylorEstimator
from samplics.utils.types import PopParam

estimator = TaylorEstimator(PopParam.prop)
estimator.estimate(
    y=df["ALT_elevated"],
    samp_weight=df["WTMEC2YR"],
    stratum=df["SDMVSTRA"],
    psu=df["SDMVPSU"],
)
```

For regression in `statsmodels` with survey correction:
```python
import statsmodels.formula.api as smf
# statsmodels does not have full SVY support;
# use samplics for design-correct regression OR
# use R's survey package via rpy2 as a fallback
```

The preferred implementation uses `samplics` for both prevalence estimates and the weighted logistic regression. If `samplics` regression API is insufficient, `rpy2` bridging to R's `survey::svyglm` is the fallback — document which was used.

### MEC weights vs. subsample weights

| Weight | Used For | Sample |
|--------|---------|--------|
| `WTMEC2YR` | All MEC-examined participants | Full analytic sample |
| `WTSB2YR` | Environmental chemical subsample (phthalates, phenols) | 1/3 of MEC participants |
| `WTDR2D` | 24-hour dietary recall | Those who completed recall |

**Critical rule:** Never mix weight types in a single model. A model using phthalate exposure as a predictor must use `WTSB2YR`, not `WTMEC2YR`. These are separate analytic frames.

---

## 7. Weighted EDA

### What "weighted" means computationally

An unweighted prevalence is `ALT_elevated.mean()` — the proportion of rows in the dataset with elevated ALT. This equals the prevalence in a simple random sample.

A weighted prevalence accounts for the fact that each participant represents a different number of U.S. adults:
```python
weighted_prevalence = (df["ALT_elevated"] * df["WTMEC2YR"]).sum() / df["WTMEC2YR"].sum()
```

This is a Horvitz-Thompson estimator. The result represents the estimated proportion of U.S. non-institutionalized civilian adults with elevated ALT — a different (and more meaningful) quantity than the sample proportion.

### What we compute

1. **Weighted prevalence of ALT elevation** by demographic subgroups — establishing the public health burden
2. **Weighted distribution of continuous predictors** — identifying skew, outliers, range issues before modeling
3. **Spearman correlations** among biomarkers — identifying multicollinearity that will need to be handled in the regression model
4. **Conditional distributions** of key predictors by ALT status — building intuition for what the models will find

### Figures produced

| Figure | What it shows | Why it matters |
|--------|--------------|----------------|
| `fig_alt_distribution.png` | ALT distribution (raw + log) with threshold lines | Confirms right-skew, shows outcome prevalence visually |
| `fig_bmi_by_alt.png` | Weighted BMI distribution by ALT status | BMI is the expected dominant predictor — confirms data quality |
| `fig_correlation_heatmap.png` | Spearman rho among continuous biomarkers | Diagnoses collinearity for regression; AST/ALT should be ~0.7 |
| `fig_prevalence_by_race.png` | Weighted ALT prevalence by race/ethnicity | Shows demographic heterogeneity; supports race/ethnicity as confounder |

---

## 8. Inference Model: Survey-Weighted Logistic Regression

### Purpose and scope

The inference model answers: **"Which exposures are independently associated with elevated ALT at the population level, after adjusting for confounders?"**

This is a population-level, associational claim. It is not a causal claim (no randomization) and not an individual-level prediction (regression coefficients are average effects across the population, not individual scores).

### Model specification

Outcome: `ALT_elevated` (binary)

Predictors grouped by role:

| Role | Variables | Coding |
|------|-----------|--------|
| **Core demographics (always adjust)** | Age (RIDAGEYR), Sex (RIAGENDR), Race/ethnicity (RIDRETH3) | Continuous; binary; 5-category dummy (ref: Non-Hispanic White) |
| **Socioeconomic** | Poverty-income ratio (INDFMPIR) | Continuous; log-transformed |
| **Anthropometric** | BMI (BMXBMI), Waist circumference (BMXWAIST) | Continuous; include one (prefer waist for visceral fat) |
| **Metabolic comorbidities** | Diabetes (DIQ010) | Binary |
| **Behavioral** | Smoking (SMQ020 current), Alcohol within-inclusion range (ALQ120Q) | Binary; continuous |
| **Biochemical biomarkers** | Total bilirubin (LBXSTB), ALP (LBXSAPSI), Albumin (LBXSAL) | Continuous (liver function panel — do not include AST, creates circular inference) |
| **Heavy metals (exposures of interest)** | Log-blood lead (LBXBPB), Log-blood cadmium (LBXBCD), Log-blood mercury (LBXTHG) | Log-transformed continuous |

**Why log-transform metals:** Blood metal distributions are highly right-skewed (lognormal). Log transformation normalizes the distribution and makes the OR interpretable as the effect per doubling of exposure (if natural log) or per order-of-magnitude increase (if log10).

**Why exclude AST as predictor:** AST (LBXSASSI) is also an aminotransferase — it measures the same underlying process as ALT. Including it as a predictor while predicting ALT elevation is near-tautological and would dominate the model, masking the environmental predictors of interest.

### Output: Odds ratio table

| Column | Content |
|--------|---------|
| `variable` | Variable name |
| `label` | Plain-English label |
| `OR` | exp(β) — odds ratio |
| `CI_lower` | Lower bound of design-correct 95% CI |
| `CI_upper` | Upper bound of design-correct 95% CI |
| `p_value` | Wald test p-value (design-correct) |
| `n_complete` | Number of non-missing observations for this predictor |

CIs are computed using Taylor-series linearization, not the naive OLS standard errors. This is what "design-correct" means.

### Interpreting ORs vs. the ML model

| Model | Estimand | Population claim | Individual claim |
|-------|---------|-----------------|-----------------|
| Weighted logistic regression | Average population-level association | Valid (with caveats about residual confounding) | Invalid — log-odds for an individual is not a probability |
| XGBoost | Individual prediction score | Invalid — not representative of U.S. population | Valid — identifies high-risk individuals in a prediction task |

This table appears in the technical report and the methods/limitations document.

---

## 9. ML Prediction Model: XGBoost + SHAP

### Why XGBoost

1. **Handles missing data natively** (learns optimal imputation direction during tree building) — critical for NHANES with its variable missingness patterns
2. **No distributional assumptions** — no need to satisfy linearity, normality of residuals, or homoscedasticity
3. **Captures non-linear effects and interactions** — relevant for complex metabolic relationships (e.g., BMI × diabetes interaction on liver stress)
4. **State-of-the-art tabular performance** — XGBoost consistently outperforms other methods on structured clinical data
5. **SHAP-compatible** — `shap.TreeExplainer` is exact (not approximate) for tree-based models, making explanations reliable

### Why NOT neural networks or deep learning

Neural networks do not improve on XGBoost for small-to-medium structured tabular datasets (N ~4,000–5,000 analytic rows). They also require substantially more hyperparameter tuning and produce less interpretable SHAP explanations. For a regulatory/safety science audience, tree-based models with exact SHAP values are preferred.

### Feature matrix

All variables from the inference model, PLUS:
- Full BIOPRO_J biochemistry panel (AST, glucose, cholesterol, triglycerides, etc.)
- All anthropometric measures from BMX_J
- Smoking pack-years if available

ALT itself and the sensitivity-analysis outcome (`ALT_elevated_40`) are excluded to prevent data leakage.

### Train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Stratification on `y` ensures the outcome class balance is preserved in both splits. This is important because ALT elevation is a minority class (~10%).

**Note on survey weights and ML:** The ML model is trained on unweighted data (each row = one participant). This is labeled "prediction-only" in all outputs. If weighted training is desired (to improve population representativeness of the model), sample weights can be passed to XGBoost's `sample_weight` parameter — but this is a stretch goal, not the core.

### Recursive Feature Elimination with Cross-Validation (RFECV)

```python
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

selector = RFECV(
    estimator=XGBClassifier(n_estimators=100, random_state=42),
    step=1,
    cv=StratifiedKFold(n_splits=5),
    scoring="roc_auc",
    n_jobs=-1,
)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_].tolist()
```

RFECV eliminates features that do not improve AUC on CV held-out folds, reducing the risk of overfitting in the final model.

### SHAP explanations

SHAP (SHapley Additive exPlanations) decomposes a model's prediction for any individual into additive contributions from each feature. For tree ensembles, `TreeExplainer` computes exact SHAP values in polynomial time.

**Global summary (beeswarm plot):**
- X-axis: SHAP value (positive = pushes toward elevated ALT, negative = pushes away)
- Y-axis: Each feature (sorted by mean |SHAP|)
- Color: Feature value (red = high, blue = low)
- Interpretation: Shows which features matter most overall AND whether high values increase or decrease risk

**Individual waterfall plots:**
- Show the feature contributions for a single individual, starting from the model's average prediction and adding/subtracting each feature's contribution to reach the final prediction
- Two examples: one high-risk individual (top decile of predicted probability) and one low-risk (bottom decile) — illustrates how the model individuates risk

**SHAP vs. the OR table:**
- SHAP rankings reflect the model's learned relationships in this dataset (not necessarily causal)
- OR table reflects population-level associations under the survey design
- Qualitative convergence between the two (same top predictors) strengthens confidence in findings
- Divergence (different top predictors) must be explained in the methods section

---

## 10. LLM-Assisted Reporting Layer

### Architecture

`06_reports.py` uses the Anthropic API to draft structured report sections from tabular inputs. The workflow:

```
Structured data (ORs, SHAP, AUC, prevalence)
        ↓
Prompt engineering (system + user prompt)
        ↓
Claude API call (claude-sonnet-4-6)
        ↓
Draft report text
        ↓
Human review + edit
        ↓
Final saved report
```

### What the LLM does and does not do

| Task | LLM does | Human must verify |
|------|---------|-------------------|
| Draft narrative around OR values | Yes | Directionality of interpretation, no overclaiming |
| Write methods section prose | Yes | Statistical accuracy, citation correctness |
| Generate executive summary bullets | Yes | Audience appropriateness |
| Make causal claims | No (prompt explicitly prohibits) | Check that no causal language slipped in |
| Interpret SHAP values | Partial (explains what high SHAP means) | Whether the feature direction makes biological sense |
| Choose what to highlight | Yes | Whether highlighted findings are scientifically most important |

### Prompt structure

Each report call uses:
1. **System prompt:** Establishes role (epidemiologist/statistician), prohibits causal language, specifies audience, sets tone
2. **User prompt:** Provides structured data tables + instructions for specific sections
3. **Temperature:** 0.3 (lower temperature for factual, reproducible outputs)

### Three report audiences

**`technical.md` — R&D Scientists**
- Full statistical methods description
- Complete OR table (all predictors)
- AUC, calibration, confusion matrix
- SHAP global summary interpretation
- Statistical caveats (survey design, complete-case analysis, no causation)
- R&D actionability: which exposures to prioritize for mechanistic follow-up

**`safety_summary.md` — Medical Safety / Consumer Toxicology**
- One-page, plain-language summary
- Weighted prevalence numbers ("approximately X% of U.S. adults have elevated ALT")
- Top 3–5 risk factors from SHAP + OR
- Exposure-response framing (higher cadmium → higher odds)
- What this means for ingredient safety assessment
- No p-values — effect sizes and CIs only

**`methods_limitations.md` — Regulatory / QA Review**
- Detailed justification of each methodological choice
- Survey design rationale (why design weights matter, how applied)
- Missing data approach and limitations
- ML model scope boundaries (prediction-only)
- Exclusion criterion justification
- Sensitivity analysis results
- Scope of inference (U.S. civilian non-institutionalized adults, 2017–2018 only)

---

## 11. Correctness Rules

These four rules are baked into code comments in every relevant file. Violating any one of them produces results that are wrong — not just suboptimal, but incorrect.

### Rule 1: SEQN is the only merge key

**Why:** SEQN is the unique participant identifier that NHANES guarantees to be stable and unique within a cycle. Any other merge attempt (e.g., row-position join) will silently misalign data.

**Enforced in:** `02_build_dataset.py` — every `pd.merge()` call specifies `on="SEQN"`, `how="left"`, and is followed by an assert on row count.

### Rule 2: Recode missing codes BEFORE any analysis

**Why:** NHANES uses numeric sentinel values for refused/don't-know responses. If not recoded, these become valid numeric observations in models. A participant who "refused" (code 77) on a smoking question appears to smoke 77 cigarettes — a catastrophic misclassification.

**Enforced in:** `02_build_dataset.py` — a dedicated `recode_sentinels(df, variable, codes)` function is called on every affected variable before the function proceeds to any recoding or analysis.

### Rule 3: Use survey weights for ALL population estimates and regression

**Why:** NHANES is a stratified multistage probability sample. Ignoring weights produces biased prevalence estimates and artificially narrow confidence intervals (because within-PSU correlation inflates the effective sample size). The ML model may be unweighted, but must be explicitly labeled as prediction-only.

**Enforced in:** `03_eda_weighted.py` and `04_inference.py` — no `mean()`, `value_counts()`, or regression call operates without the design object. Comments label where unweighted computation is used and why it is acceptable (e.g., XGBoost feature matrix construction).

### Rule 4: Do NOT drop incomplete rows before constructing the survey design object

**Why:** The design weights were calibrated to the full examined sample. Dropping rows with missing predictor values before constructing the design object changes the sample in a way the weights cannot correct. Dropping rows after the design object is constructed, within a specific model's complete-case analysis, is acceptable — the design object itself remains intact.

**Enforced in:** `02_build_dataset.py` — the analytic table saved to parquet retains all rows with `WTMEC2YR > 0` and valid outcome; predictor missingness is preserved as NaN and handled per-model.

---

## 12. Deliverables Map

| Deliverable | File Path | Produced By | Description |
|-------------|-----------|------------|-------------|
| Analytic dataset | `/data/processed/analytic_table.parquet` | `02_build_dataset.py` | Merged, cleaned, outcome-defined, exclusions applied |
| Codebook | `/data/processed/codebook.csv` | `02_build_dataset.py` | Variable → label → unit → module |
| Weighted prevalence table | `/reports/weighted_prevalence_table.csv` | `03_eda_weighted.py` | Prevalence by demographic subgroups |
| ALT distribution figure | `/figures/fig_alt_distribution.png` | `03_eda_weighted.py` | Raw + log ALT with thresholds |
| BMI by ALT figure | `/figures/fig_bmi_by_alt.png` | `03_eda_weighted.py` | BMI distribution by outcome |
| Correlation heatmap | `/figures/fig_correlation_heatmap.png` | `03_eda_weighted.py` | Spearman rho among biomarkers |
| Prevalence by race | `/figures/fig_prevalence_by_race.png` | `03_eda_weighted.py` | Weighted prevalence by RIDRETH3 |
| OR inference table | `/reports/inference_OR_table.csv` | `04_inference.py` | Design-correct ORs + CIs |
| Forest plot | `/figures/fig_forest_plot_OR.png` | `04_inference.py` | Visual OR table |
| ML metrics | `/reports/ml_metrics.json` | `05_predict_ml.py` | AUC, precision-recall, confusion matrix |
| Selected features | `/reports/selected_features.txt` | `05_predict_ml.py` | RFECV-selected predictor list |
| SHAP summary | `/figures/fig_shap_summary.png` | `05_predict_ml.py` | Global beeswarm |
| SHAP importance | `/figures/fig_shap_importance.png` | `05_predict_ml.py` | Mean |SHAP| bar chart |
| SHAP waterfalls (2) | `/figures/fig_shap_waterfall_*.png` | `05_predict_ml.py` | Individual-level explanations |
| SHAP rankings | `/reports/shap_rankings.csv` | `05_predict_ml.py` | Feature, mean_abs_shap |
| Technical report | `/reports/technical.md` | `06_reports.py` + human | Full methods + results for R&D |
| Safety summary | `/reports/safety_summary.md` | `06_reports.py` + human | Plain-language for Medical Safety |
| Methods/limitations | `/reports/methods_limitations.md` | `06_reports.py` + human | Regulatory-level methods documentation |
| AI usage appendix | `/reports/ai_usage_appendix.md` | Human-written | Transparency on AI assistance |
| Main notebook | `/notebooks/main.ipynb` | Human-written | End-to-end narrative |
| README | `/README.md` | Human-written | Project entry point |
| One-command runner | `/run_all.sh` | Human-written | `bash run_all.sh` → all outputs |

---

## 13. What's New vs. The Field

### Positioning paragraph

> Recent 2024–25 NHANES machine-learning studies predict liver outcomes (ALT elevation, MASLD, NAFLD) using XGBoost, random forest, and neural networks with AUCs of 0.75–0.90 and SHAP-based explanations. However, virtually all treat NHANES as a simple random sample: prevalence estimates are unweighted proportions, confidence intervals are from ordinary logistic regression, and model AUCs are reported without noting that the training sample is not representative of the U.S. population. This project reproduces the predictive approach while adding (1) survey-design-correct prevalence estimation, (2) Taylor-series-linearization CIs for population-level inference, and (3) an explicit separation between population inference (weighted regression) and individual prediction (XGBoost) — a methodological distinction the prior literature blurs. The result is a transferable pipeline for exposome-wide risk screening that produces both inference-valid population estimates and interpretable individual predictions from the same dataset.

### Representative prior literature (to cite in reports)

- Younossi et al. (2022–2024) — MASLD/NAFLD prevalence from NHANES (methodology to contrast against)
- Liang et al. (2023) — XGBoost for liver disease prediction in NHANES
- CDC NCHS Data Briefs — authoritative weighted prevalence benchmarks to validate against

### What this project adds that is genuinely new

1. Survey-correct inference + ML prediction in the same pipeline (dual-method design)
2. Exposome-wide panel including heavy metals alongside metabolic biomarkers
3. Explicit labeling of which outputs are population-level (inference) vs. individual-level (prediction)
4. Three stakeholder-specific report formats from the same underlying analysis

---

## 14. Limitations & Future Work

### Limitations of the core analysis

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Cross-sectional design | Cannot establish temporality; cannot infer causation | All claims are associational; explicit language in every report |
| Complete-case analysis | Participants missing any predictor excluded per model; may introduce selection bias | Document missingness rates; conduct sensitivity analysis on imputed data as stretch |
| Single survey cycle (2017–2018) | Cannot assess trends over time | Explicitly scoped; future work = pooling 2015–2020 cycles |
| Self-reported behavioral variables | Alcohol, smoking subject to social desirability bias; cadmium from smoking may be underestimated | Cadmium biomarker (LBXBCD) is objective; use as primary smoking-exposure proxy |
| Exclusion of viral hepatitis | May underestimate overall ALT burden; limits generalizability to hepatitis-endemic contexts | Stated scope; appropriate for xenobiotic-focused analysis |
| ML model trained on unweighted data | Predictions are not representative of U.S. population probabilities | Explicitly labeled as prediction-only throughout |

### Future work

1. **Multiple imputation:** Replace complete-case analysis with `miceforest` or `IterativeImputer` to preserve more observations and reduce selection bias
2. **Multi-cycle pooling:** Pool 2015–2018 (NHANES cycles G + H + I + J) using combined 4-year weights (`WTMEC4YR` equivalent) for larger N and trend analysis
3. **Environmental-chemical panel:** Add phthalates (PHTHTE_J) and phenols/parabens (EPH_J) using the `WTSB2YR` subsample weight — see Stretch Goal section in ROADMAP.md
4. **Causal inference:** Apply g-computation or targeted maximum likelihood estimation (TMLE) for specific exposure-outcome pairs identified in the associational analysis
5. **Longitudinal validation:** Replicate key findings in NHANES continuous (2019–2020) or a clinical cohort dataset

### Stretch Goal Detail: Environmental-Chemical Panel

**Modules to add:** PHTHTE_J (urinary phthalate metabolites), EPH_J (urinary phenols and parabens)

**Critical design rule:** These analytes were measured in a 1/3 environmental subsample. The subsample weight is `WTSB2YR`, not `WTMEC2YR`. To analyze these variables correctly:
1. Filter to participants with `WTSB2YR > 0`
2. Build a separate survey design object using `WTSB2YR` as the weight
3. Run inference and ML on this subsample only, with `WTSB2YR` as the weight
4. Never merge the full-sample model (WTMEC2YR) and the subsample model (WTSB2YR) results into a single model — the weights are not interchangeable

This yields a genuinely "exposomics" angle: a panel of environmental chemicals alongside metabolic biomarkers, screened with survey-correct inference.

---

## 15. AI Usage Appendix

This section documents exactly where AI assistance (Claude, Anthropic API) was used in building this project and where human judgment overrode or supplemented it. This transparency is both scientifically important and a competitive differentiator in a field where AI usage is often undisclosed.

### Where AI was used

| Task | AI role | Human verification required |
|------|---------|---------------------------|
| NHANES codebook variable labeling | Claude batch-translated 50+ variable codes (e.g., `LBXSATSI`) to plain-English labels from CDC codebook HTML | Each label verified against official CDC documentation at nhanes.cdc.gov |
| Report drafting (`06_reports.py`) | Claude drafted narrative sections given structured OR and SHAP inputs | Every OR value, CI, and AUC cross-checked against source data; causal language removed |
| Methods section prose | Claude drafted first-pass methods description from structured inputs | Statistical accuracy of survey design description verified against Lumley (2004) and CDC documentation |
| Executive summary bullets | Claude prioritized findings from the OR table | Ranking cross-checked against biological plausibility and SHAP consistency |

### Where human judgment overrode AI

| Decision | Why human judgment was required |
|----------|-------------------------------|
| Exclusion criterion thresholds (ALQ510 > 14 male / > 7 female) | Required domain knowledge of NIAAA definitions and published NHANES liver studies |
| Sex-specific ALT thresholds (56 U/L male / 33 U/L female) | Clinical judgment call; required literature review of AASLD guidelines; AI initially suggested 40 U/L unisex |
| Decision to exclude AST as predictor | Statistical reasoning about circular inference; AI included it in initial model spec |
| Survey weight type selection (MEC vs. subsample) | Required careful reading of NHANES analytic guidelines; critical correctness issue |
| Causal language removal from reports | AI occasionally slipped "causes" or "leads to" into draft text; removed and replaced with "associated with" |
| Scope framing ("exposomics vs. environmental epidemiology") | Positioning judgment for the Kenvue application context |

### How to reproduce without LLM

If `ANTHROPIC_API_KEY` is not set, `06_reports.py` falls back to template-based report stubs stored in `/src/report_templates/`. These templates contain all section headers and placeholder strings (e.g., `{OR_TABLE}`, `{AUC}`) that can be filled manually. The scientific content of the pipeline (data, inference, ML) is entirely reproducible without the API.

### Honest assessment of AI limitations in this project

The LLM excelled at: narrative prose generation, plain-language translation of statistical results, formatting consistency across three reports.

The LLM required human correction for: statistical precision (occasionally rounded CIs incorrectly), causal framing (required explicit prohibition in system prompt), threshold/cutoff choices (required domain knowledge not in training data), and any claim about specific NHANES variable availability (hallucinated variable names once — always verify against actual downloaded data).
