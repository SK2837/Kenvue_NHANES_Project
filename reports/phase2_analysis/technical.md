# NHANES 2017–2018: Population-Level Drivers and Individual-Risk Prediction for Elevated ALT

## 1. Background & Objective
The National Health and Nutrition Examination Survey (NHANES) 2017-2018 provides a nationally representative dataset to investigate population-level drivers and individual-risk prediction for elevated alanine transaminase (ALT), a primary marker of liver injury. This analysis identifies independent risk factors for elevated ALT (> 40 U/L) using two complementary approaches: survey-weighted logistic regression for causal inference and XGBoost machine learning for individual risk prediction.

## 2. Methods
The NHANES 2017-2018 design uses complex, multistage probability sampling. All estimates apply survey weights (WTMEC2YR) with Taylor-series linearization to produce design-correct confidence intervals representative of the U.S. civilian adult population.

**Model evolution:** The initial 9-predictor model (age, sex, poverty-income ratio, BMI, diabetes, smoking, blood lead/cadmium/mercury) was extended through a hypothesis-driven search across 13 candidate variables including metabolic, behavioral, and social predictors. Two new significant predictors were identified — waist circumference and triglycerides — and incorporated into the final 10-predictor model. Waist circumference replaces BMI as the anthropometric measure because it more directly captures visceral adiposity, the fat depot most implicated in hepatic steatosis.

An XGBoost ML pipeline with recursive feature elimination (RFECV) trained on the full blood panel (31 selected features) was developed in parallel for individual prediction.

## 3. Results

### 3.1 Prevalence
Weighted prevalence of elevated ALT: **5.2%** (N = 3,543). Notable subgroup differences: Mexican Americans 8.9%, Non-Hispanic Asian 7.0%; adults 18–40: 8.7% vs. adults 60+: 2.7%.

### 3.2 Survey-Weighted Logistic Regression — Updated 10-Predictor Model

| Variable | OR | 95% CI | p-value | Significant |
| --- | --- | --- | --- | --- |
| Age (years) | 0.960 | [0.939, 0.982] | 0.005 | ✓ |
| Male sex | 2.025 | [0.937, 4.377] | 0.065 | — |
| Poverty-income ratio | 1.015 | [0.864, 1.192] | 0.823 | — |
| **Waist circumference (cm)** | **1.027** | **[1.009, 1.044]** | **0.012** | **✓ NEW** |
| Diabetes | 1.975 | [0.922, 4.232] | 0.070 | — |
| Ever smoker | 0.641 | [0.376, 1.095] | 0.086 | — |
| **Triglycerides (log)** | **1.773** | **[1.160, 2.710]** | **0.018** | **✓ NEW** |
| log(Blood Lead) | 1.373 | [0.837, 2.252] | 0.160 | — |
| log(Blood Cadmium) | 0.937 | [0.661, 1.328] | 0.651 | — |
| log(Blood Mercury) | 1.042 | [0.887, 1.223] | 0.542 | — |

### 3.3 Extended Hypothesis Test Summary (13 Variables)

| Variable | Category | Significant in Any Model? |
| --- | --- | --- |
| Age (years) | Demographics | ✓ Yes (all models) |
| Waist circumference | Anthropometrics | ✓ Yes — NEW |
| Triglycerides (log) | Clinical | ✓ Yes — NEW |
| Diabetes | Clinical | ✓ Yes (original model) |
| Male sex | Demographics | ✓ Yes (original model) |
| Alcohol, College education | Behavioral/Social | No |
| Blood lead, cadmium, mercury | Environmental | No — across all models |

### 3.4 ML Model Performance

| Metric | Value |
| --- | --- |
| AUC-ROC | 0.9731 |
| PR-AUC | 0.7926 |
| Sensitivity | 0.9259 |
| Specificity | 0.9221 |
| Test N | 709 (54 events) |

Top SHAP features: AST, GGT, Age, BMI, **Triglycerides**, **Waist circumference**, Blood Mercury, Albumin, Glucose, Globulin. Triglycerides and waist circumference appear in both the SHAP top-10 and the updated epidemiological model — cross-validation of findings across methods.

## 4. Discussion
The updated 10-predictor model confirms that **metabolic syndrome markers** are the dominant modifiable predictors of elevated ALT in U.S. adults:

- **Waist circumference** (OR = 1.027 per cm, p = 0.012): Each additional centimeter increases adjusted odds of elevated ALT by 2.7%. Visceral fat drives hepatic lipid accumulation (NAFLD pathway), independent of other metabolic factors.

- **Triglycerides** (OR = 1.773 per log unit, p = 0.018): A person with triglycerides of 200 mg/dL has approximately 27% higher adjusted odds of elevated ALT than someone at 100 mg/dL, after full adjustment. Hypertriglyceridemia is a direct consequence of excess hepatic fat synthesis.

- **Mediation insight:** When both waist and triglycerides enter the model, the ORs for male sex (2.643 → 2.025) and diabetes (2.197 → 1.975) attenuate and lose significance. This is mechanistically coherent — both conditions elevate visceral fat and triglycerides, which are the proximate drivers of liver stress.

- **Heavy metals null finding:** Blood lead, cadmium, and mercury remained non-significant across all three model specifications, providing consistent evidence against environmental metal toxicity as a driver of ALT elevation in the general U.S. adult population at current exposure levels.

## 5. Statistical Notes & Caveats
- All estimates are survey-weighted (WTMEC2YR) with Taylor-series CIs
- Events per variable (EPV): updated model EPV ≈ 18.5 — adequate for stable estimation
- Extended hypothesis test was exploratory; no multiple-testing correction applied
- Cross-sectional design: associations only, no causal claims
- Heavy metal exposure measured at single time point; chronic exposure not captured
- ML model trained on unweighted data; predictive performance may differ in weighted projection
