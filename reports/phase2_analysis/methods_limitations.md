**Methods, Statistical Design, and Limitations — NHANES 2017–2018 ALT Analysis**

### 1. Survey Design & Weighting

The National Health and Nutrition Examination Survey (NHANES) 2017-2018 employs a complex, multistage probability sampling design. To account for this design and ensure representative estimates of the U.S. population, we utilize the provided survey weights (WTMEC2YR). These weights adjust for non-response, non-coverage, and the unequal probabilities of selection. The use of WTMEC2YR allows our analysis to be generalizable to the broader U.S. adult population. To accurately estimate variances and construct confidence intervals (CIs), we employ Taylor-series linearization, which provides design-correct CIs essential for valid statistical inference in complex surveys.

### 2. Outcome Definition

The primary outcome is elevated ALT (> 40 U/L), using a unisex threshold so that biological sex can be included as a covariate without threshold-outcome interaction. A sensitivity analysis using sex-specific AASLD thresholds (33 U/L for women, 45 U/L for men) was conducted to assess robustness.

### 3. Exclusion Criteria & Sample Flow

Starting from the NHANES 2017-2018 examined subsample (adults 18+), we exclude individuals with hepatitis B or C to isolate ALT elevations from non-viral causes. Final analytic sample: N = 3,543.

### 4. Inference Model Specification & Assumptions

#### Original model (9 predictors)
Initial specification: age, sex, poverty-income ratio, BMI, diabetes status, smoking history, and log-transformed blood lead, cadmium, and mercury.

#### Extended hypothesis test
We systematically tested 4 additional candidate predictors: waist circumference, log(triglycerides), log(weekly alcohol drinks), and college education. Waist and log(triglycerides) emerged as statistically significant (p < 0.05), while alcohol and college education were not.

#### Updated primary model (10 predictors — final)
Based on the extended test, we updated the primary model to replace BMI with waist circumference (a superior proxy for visceral adiposity) and added log(triglycerides) as a new predictor. This 10-predictor specification is the primary model used for all inference, individual risk estimation, and hypothesis comparisons.

**Predictor list (final model):** age, sex, poverty-income ratio, waist circumference (cm), diabetes, ever smoker, log(triglycerides), log(blood lead), log(blood cadmium), log(blood mercury).

Key model assumptions: linearity in the log-odds, no major unmeasured confounders, missing data missing at random (MAR) conditional on observed covariates.

### 5. ML Model — Scope and Boundaries

An XGBoost model with Recursive Feature Elimination and Cross-Validation (RFECV) was trained on the full available blood panel to select the 31 most informative features. This model is optimized for individual prediction accuracy (AUC-ROC = 0.9731) rather than causal inference. It is trained on unweighted data to maximize predictive performance. The top features include liver enzymes (AST, GGT), age, triglycerides, and waist circumference — consistent with the epidemiologic model findings. AST and GGT dominate SHAP importance because they are correlated outcomes, not independent causes of elevated ALT.

### 6. Missing Data Approach

Missing data is assumed to be MAR conditional on observed covariates. Survey weights partially account for non-response bias. The logistic regression uses complete-case analysis (2,923 of 3,543 rows after listwise deletion). The ML model imputes missing values using XGBoost's native handling.

### 7. Key Limitations

1. **Cross-sectional design**: Cannot establish causation. All associations are observational.
2. **Single survey cycle**: Cannot capture temporal trends in exposure-outcome relationships.
3. **Single-point metal measurement**: Blood lead, cadmium, and mercury reflect recent rather than cumulative exposure; chronic effects may differ.
4. **Extended hypothesis testing**: The test of 4 additional variables was exploratory. No multiple-testing correction was applied; the two significant findings (waist, triglycerides) should be interpreted as hypothesis-generating rather than confirmatory without replication.
5. **Multicollinearity**: BMI and waist circumference are highly correlated (r ≈ 0.85); glucose, triglycerides, and diabetes share a metabolic pathway. Inclusion of all simultaneously inflates standard errors.
6. **Generalizability**: Findings reflect the U.S. civilian non-institutionalized adult population in 2017–2018 only.

### 8. What These Results Cannot Claim

- Cannot establish that waist circumference or triglycerides *cause* elevated ALT (cross-sectional)
- Cannot be generalized to non-U.S. populations or clinical patients
- Cannot confirm that reducing waist or triglycerides will lower ALT (intervention design required)
- ML model predictions should not replace clinical laboratory testing or medical judgment
