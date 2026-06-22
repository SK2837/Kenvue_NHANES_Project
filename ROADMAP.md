# NHANES Exposome-Wide Liver-Injury Screen — Two-Day Execution Roadmap

## Project Overview

This pipeline builds a reproducible, survey-design-correct analysis of elevated ALT (liver injury) in the U.S. adult population using NHANES 2017–2018. It separates **population-level inference** (survey-weighted logistic regression with design-correct confidence intervals) from **individual-level prediction** (XGBoost + SHAP), a distinction the published literature routinely blurs. The output is three stakeholder-specific reports and a fully reproducible Python pipeline that serves as a transferable risk-assessment method for any orally-dosed compound's hepatic safety profile.

---

## Prerequisites

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Required Libraries (requirements.txt)
```
pandas>=2.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
samplics>=0.4          # survey-weighted regression
xgboost>=2.0
scikit-learn>=1.3
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
pyreadstat>=1.2        # read .XPT files
requests>=2.31
anthropic>=0.25        # LLM report drafting
jupyter>=1.0
black
flake8
mypy
pytest
```

### API Keys
- `ANTHROPIC_API_KEY` in environment or `.env` file — needed only for `06_reports.py`

### External Data
- All NHANES .XPT files downloaded automatically by `01_download.py`; no manual download required
- Expected download size: ~15 MB total; files cached in `/data/raw/`

---

## Day 1 — Data Acquisition, Cleaning, and Weighted EDA

**Goal:** End the day with a clean analytic dataset (`/data/processed/analytic_table.parquet`) and a set of weighted EDA figures that confirm the pipeline is survey-design-correct.

---

### Block 1.1 — Repo Skeleton & `requirements.txt`

**Duration:** ~20 min

**Tasks:**
- [ ] Create directory structure:
  ```
  /data/raw/
  /data/processed/
  /src/
  /notebooks/
  /reports/
  /figures/
  ```
- [ ] Write `requirements.txt` (see Prerequisites above)
- [ ] Initialize `.gitignore` (ignore `/data/raw/`, `.env`, `__pycache__`, `.venv`)
- [ ] Create stub files for `src/01` through `src/06`

**Deliverable:** Repo skeleton exists; `pip install -r requirements.txt` runs clean.

---

### Block 1.2 — `src/01_download.py`: Pull NHANES .XPT Files

**Duration:** ~30 min

**Tasks:**
- [ ] Write a `download_xpt(module_code, year="2017-2018")` function that:
  - Constructs the CDC URL: `https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{MODULE_CODE}.XPT`
  - Checks if file already exists locally before downloading (cache-first)
  - Saves to `/data/raw/{MODULE_CODE}.XPT`
  - Logs file size and row count after download
- [ ] Download all required modules:

  | Module Code | Purpose |
  |-------------|---------|
  | DEMO_J      | Demographics, survey weights, strata, PSU |
  | BMX_J       | BMI, waist circumference |
  | BIOPRO_J    | Full biochemistry panel including ALT (outcome) |
  | PBCD_J      | Blood lead, cadmium, mercury |
  | ALQ_J       | Alcohol use |
  | DIQ_J       | Diabetes diagnosis |
  | SMQ_J       | Smoking (cadmium confounder) |
  | HEPB_S_J    | Hepatitis B serology (exclusion criterion) |
  | HEPC_J      | Hepatitis C serology (exclusion criterion) |

- [ ] Print row counts for each module after download — spot-check against CDC documentation (DEMO_J should have ~9,254 rows)

**Correctness rule enforced:** Files cached locally; script is idempotent (re-running does not re-download).

**Deliverable:** All `.XPT` files present in `/data/raw/`.

---

### Block 1.3 — `src/02_build_dataset.py`: Merge, Recode, Define Outcome

**Duration:** ~90 min — this is the most critical block; do not rush it.

**Tasks:**

#### Step A: Load and left-join all modules onto DEMO_J
- [ ] Load `DEMO_J` as the base frame (all participants)
- [ ] Left-join each additional module on `SEQN` (never inner-join — this would drop participants silently)
- [ ] After each join, assert row count equals DEMO_J row count (write a test)

#### Step B: Recode missing/refused/don't-know codes to NaN
NHANES uses numeric sentinel values for non-response. These MUST be recoded before any analysis.

| Code | Meaning | Variables affected |
|------|---------|-------------------|
| 7    | Refused  | Single-digit items |
| 77   | Refused  | Two-digit items |
| 777  | Refused  | Three-digit items |
| 9    | Don't know | Single-digit |
| 99   | Don't know | Two-digit |
| 999  | Don't know | Three-digit |

- [ ] Apply recoding systematically to all non-continuous variables
- [ ] Verify: check that `LBXSATSI` (ALT) has no values of 7/9/99 (it is continuous — sentinel is `.` in XPT, already NaN on load)
- [ ] Document which variables had sentinel values recoded (log count of recoded cells)

#### Step C: Define binary ALT outcome
```
ALT_elevated = 1  if:
    LBXSATSI > 56  AND  RIAGENDR == 1  (male)
    OR
    LBXSATSI > 33  AND  RIAGENDR == 2  (female)
ALT_elevated = 0  otherwise
ALT_elevated = NaN  if LBXSATSI is NaN
```
- [ ] Create `ALT_elevated` column
- [ ] Create `ALT_elevated_40` column for sensitivity analysis (threshold = 40 U/L for both sexes)
- [ ] Log raw prevalence (unweighted) as a sanity check — expect ~8–12% elevated

#### Step D: Apply exclusion criteria
Apply sequentially and log N dropped at each step:

| Step | Criterion | Variable | Direction |
|------|-----------|----------|-----------|
| 1 | Adults 18+ only | `RIDAGEYR >= 18` | Keep |
| 2 | Examined subsample only | `WTMEC2YR > 0` | Keep |
| 3 | Non-missing ALT | `LBXSATSI not NaN` | Keep |
| 4 | Hep B negative | `LBXHBS != 1` (surface antigen negative) | Keep |
| 5 | Hep C negative | `LBXHCR != 1` (antibody negative) | Keep |
| 6 | Non-excessive alcohol | `ALQ510 <= 14` (male) or `ALQ510 <= 7` (female) | Keep |

- [ ] Log N at each exclusion step — print a flow table
- [ ] Confirm final analytic N is approximately 4,000–5,500

#### Step E: Save analytic table
- [ ] Save to `/data/processed/analytic_table.parquet`
- [ ] Also save a codebook CSV listing each variable, its source module, and a plain-English label

**Correctness rules enforced:**
1. SEQN is the only merge key; left-joins preserve all DEMO_J rows
2. Missing codes recoded BEFORE any analysis
3. WTMEC2YR kept for all records (used downstream in survey design)
4. Rows NOT dropped for missingness until after survey design object is built

**Deliverable:** `/data/processed/analytic_table.parquet` with correct row count and no sentinel values.

---

### Block 1.4 — `src/03_eda_weighted.py`: Weighted EDA + Survey Design Object

**Duration:** ~60 min

**Tasks:**

#### Step A: Build survey design object
Using `samplics` or `statsmodels`:
```python
# Key parameters
weights   = "WTMEC2YR"
strata    = "SDMVSTRA"
psu       = "SDMVPSU"
```
- [ ] Construct design object and confirm it replicates CDC's published weighted prevalence estimates for at least one demographic variable (e.g., % male should be ~48–51%)

#### Step B: Weighted prevalence table
- [ ] Compute weighted prevalence of `ALT_elevated` overall and by:
  - Sex (RIAGENDR)
  - Age group (18–40, 41–60, 60+)
  - Race/ethnicity (RIDRETH3)
  - BMI category (<25, 25–30, 30+)
  - Diabetes status (DIQ010)
- [ ] Save as `/reports/weighted_prevalence_table.csv`

#### Step C: Distribution figures
Save all figures to `/figures/`:
- [ ] Histogram of ALT (raw, log-transformed) with threshold lines — `fig_alt_distribution.png`
- [ ] Weighted BMI distribution by ALT-elevated status — `fig_bmi_by_alt.png`
- [ ] Heatmap: Spearman correlations among continuous biomarkers (BIOPRO panel + heavy metals) — `fig_correlation_heatmap.png`
- [ ] Bar chart: weighted prevalence of ALT elevation by race/ethnicity — `fig_prevalence_by_race.png`

#### Step D: Spot-check survey design correctness
- [ ] Compute weighted mean BMI — should be ~28–30 kg/m² for U.S. adults
- [ ] Compare to CDC NCHS Data Brief estimates; flag any >5% discrepancy

**Deliverable:** Clean prevalence table, 4 figures, survey design object reused in `04_inference.py`.

---

### Day 1 Checkpoint

Before closing Day 1, verify:
- [ ] `/data/processed/analytic_table.parquet` exists and loads cleanly
- [ ] Row count logged and matches expected range
- [ ] No sentinel values remain in the analytic table (run `df.isin([7, 77, 9, 99]).sum()`)
- [ ] All 4 EDA figures saved to `/figures/`
- [ ] Weighted ALT prevalence matches published NHANES estimates (~8–12% by AASLD thresholds)

---

## Day 2 — Inference, ML, Reports, and Packaging

**Goal:** End the day with all deliverables present, pipeline runs end-to-end with one command, three stakeholder reports drafted.

---

### Block 2.1 — `src/04_inference.py`: Survey-Weighted Logistic Regression

**Duration:** ~90 min

**Tasks:**

#### Step A: Model specification
Independent variables entering the model:
- **Demographics:** age (continuous), sex, race/ethnicity (reference: Non-Hispanic White), poverty-income ratio
- **Anthropometrics:** BMI, waist circumference
- **Metabolic:** fasting glucose or diabetes status, total bilirubin, albumin, alkaline phosphatase
- **Exposures of interest:** log-transformed blood lead (LBXBPB), cadmium (LBXBCD), mercury (LBXTHG)
- **Behavioral:** smoking status (SMQ020), alcohol use (ALQ010)
- **Comorbidities:** BMI (already above), diabetes status

#### Step B: Run survey-weighted logistic regression
```python
# samplics API sketch
from samplics.estimation import TaylorEstimator
# OR use statsmodels with survey correction
```
- [ ] Fit model; extract odds ratios (exp(β)) and 95% CIs using Taylor-series linearization
- [ ] Test model fit (Wald test for each predictor)
- [ ] Save full OR table to `/reports/inference_OR_table.csv`

#### Step C: Sensitivity analysis
- [ ] Re-run with `ALT_elevated_40` (40 U/L unisex threshold)
- [ ] Compare ORs — note any qualitative differences

#### Step D: Forest plot
- [ ] Plot ORs with CIs for all predictors — `fig_forest_plot_OR.png`
- [ ] Color heavy-metal predictors distinctly from demographics

**Correctness rules enforced:** All estimates use survey weights + design correction; CIs labeled as design-correct; model interpretation section explicitly states these are population-level associations, not causal claims.

**Deliverable:** `/reports/inference_OR_table.csv` + forest plot figure.

---

### Block 2.2 — `src/05_predict_ml.py`: XGBoost + SHAP

**Duration:** ~120 min

**Tasks:**

#### Step A: Feature matrix construction
- [ ] Use all variables from Block 2.1 PLUS the full BIOPRO_J biochemistry panel as features
- [ ] Log-transform right-skewed biomarkers (lead, cadmium, mercury, ALT, AST, ALP)
- [ ] Do NOT include `ALT_elevated_40` (alternative outcome) as a feature — data leakage

#### Step B: Train/test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### Step C: RFE feature selection
- [ ] Use `sklearn.feature_selection.RFECV` with XGBoost estimator and 5-fold CV
- [ ] Log selected feature count vs. full feature count
- [ ] Save selected features list to `/reports/selected_features.txt`

#### Step D: XGBoost hyperparameter tuning
- [ ] Grid search over:
  - `n_estimators`: [100, 300, 500]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `subsample`: [0.8, 1.0]
- [ ] Use 5-fold stratified CV; optimize on AUC-ROC

#### Step E: Evaluate on held-out test set
- [ ] AUC-ROC (primary metric)
- [ ] Precision-Recall AUC
- [ ] Calibration plot (reliability diagram)
- [ ] Confusion matrix at optimal threshold (Youden's J)
- [ ] Save all metrics to `/reports/ml_metrics.json`

#### Step F: SHAP explanations
- [ ] Compute SHAP values on test set using `shap.TreeExplainer`
- [ ] Global summary plot (beeswarm) — `fig_shap_summary.png`
- [ ] Feature importance bar chart (mean |SHAP|) — `fig_shap_importance.png`
- [ ] Two waterfall plots: one high-risk individual, one low-risk — `fig_shap_waterfall_high.png`, `fig_shap_waterfall_low.png`
- [ ] Save SHAP ranking table (feature, mean_abs_shap) to `/reports/shap_rankings.csv`

**Correctness rules enforced:** ML model explicitly labeled "prediction-only, not population inference" in all outputs; trained on unweighted data with this limitation documented.

**Deliverable:** Trained model, metrics JSON, 4 SHAP figures, rankings CSV.

---

### Block 2.3 — `src/06_reports.py`: LLM-Assisted Stakeholder Report Drafting

**Duration:** ~60 min

**Tasks:**
- [ ] Load structured inputs:
  - OR table from `/reports/inference_OR_table.csv`
  - SHAP rankings from `/reports/shap_rankings.csv`
  - ML metrics from `/reports/ml_metrics.json`
  - Weighted prevalence table from `/reports/weighted_prevalence_table.csv`

- [ ] Call Anthropic API (`claude-sonnet-4-6`) with structured prompts to draft three reports:

  | Report | Audience | Key Content |
  |--------|---------|-------------|
  | `technical.md` | R&D scientists | Full methods, OR table, AUC, SHAP rankings, statistical caveats |
  | `safety_summary.md` | Medical Safety / regulatory | Plain-language prevalence findings, top risk factors, exposure-response language |
  | `methods_limitations.md` | Regulatory / QA | Survey design rationale, ML limitations, missing data approach, scope boundaries |

- [ ] Human-review loop: after each draft is generated, prompt user to approve or edit before saving
- [ ] Save all three reports to `/reports/`
- [ ] Log which sections were AI-drafted vs. human-edited

**Note:** If `ANTHROPIC_API_KEY` is not set, script falls back to templated stubs and logs a warning.

**Deliverable:** Three drafted report files in `/reports/`.

---

### Block 2.4 — `notebooks/main.ipynb`: Narrative Walkthrough

**Duration:** ~60 min

**Tasks:**
- [ ] Create Jupyter notebook that runs the full pipeline narratively:
  - Section 1: Problem statement and data overview
  - Section 2: Exclusion flow table and weighted prevalence
  - Section 3: Weighted EDA figures (inline)
  - Section 4: Inference model — OR table rendered as formatted table + forest plot
  - Section 5: ML model — AUC curve + SHAP summary inline
  - Section 6: Key findings narrative (2–3 paragraphs)
  - Section 7: Limitations
- [ ] Each section has a markdown cell explaining what is shown and why
- [ ] Notebook runs end-to-end with `jupyter nbconvert --to notebook --execute`

**Deliverable:** Fully executed `notebooks/main.ipynb` committed.

---

### Block 2.5 — `reports/`: Write Final Report Files

**Duration:** ~45 min

**Tasks:**
- [ ] Review AI-drafted `technical.md` — verify OR values match inference table exactly
- [ ] Review `safety_summary.md` — confirm no overclaiming (no causal language for ML predictions)
- [ ] Review `methods_limitations.md` — ensure survey design section is accurate
- [ ] Write `reports/ai_usage_appendix.md` manually:
  - Where AI was used: codebook variable labeling, report section drafting
  - Where human judgment overrode AI: exclusion criterion wording, threshold justification, statistical caveat language
  - How to reproduce without LLM: stub templates in `06_reports.py` fallback mode

**Deliverable:** All four report files complete and reviewed.

---

### Block 2.6 — Final Packaging

**Duration:** ~30 min

**Tasks:**
- [ ] Write `README.md`:
  - Project title + one-paragraph abstract
  - "What's new vs. the field" positioning paragraph (see PROJECT_DEEP_DIVE.md §13)
  - Quick-start instructions
  - Directory structure table
  - Deliverables list with file paths
  - Citation guidance
- [ ] Write `run_all.sh`:
  ```bash
  #!/bin/bash
  set -e
  python src/01_download.py
  python src/02_build_dataset.py
  python src/03_eda_weighted.py
  python src/04_inference.py
  python src/05_predict_ml.py
  python src/06_reports.py
  jupyter nbconvert --to notebook --execute notebooks/main.ipynb
  echo "Pipeline complete. See /reports/ for outputs."
  ```
- [ ] Run `flake8 src/` — resolve all E/W errors
- [ ] Run `black src/ --check` — auto-format
- [ ] Run `pytest` — all tests pass

---

### Day 2 Checkpoint

Before closing Day 2, verify every item:

**Pipeline:**
- [ ] `bash run_all.sh` completes without errors
- [ ] No hardcoded paths — all paths relative to repo root or constructed from `pathlib`

**Data:**
- [ ] `/data/processed/analytic_table.parquet` present
- [ ] Weighted ALT prevalence within expected range

**Inference:**
- [ ] `/reports/inference_OR_table.csv` — all 95% CIs are design-correct (not naive OLS)
- [ ] Heavy-metal ORs are in a plausible direction (lead/cadmium elevated OR expected)

**ML:**
- [ ] Test AUC > 0.70 (below this suggests data or feature issue — investigate before reporting)
- [ ] SHAP rankings match the pattern from the OR table directionally (sanity check)

**Reports:**
- [ ] Three stakeholder reports present and reviewed
- [ ] `ai_usage_appendix.md` complete
- [ ] No causal language in ML sections

**Notebook:**
- [ ] Executes end-to-end
- [ ] All figures render inline

---

## Scope Management

### If Day 2 runs long — what to cut
Cut in this order (ship the cleaner version rather than a broken ambitious one):

1. **Cut first:** Sensitivity analysis (`ALT_elevated_40`) — note as future work
2. **Cut second:** Full BIOPRO panel as ML features — use only the variables from inference model
3. **Cut third:** `06_reports.py` LLM drafting — write reports manually from templates
4. **Cut last:** Never cut the survey-weighted inference model — this is the core scientific contribution

A finished narrow project beats an unfinished broad one. Saying so in your writeup is itself the senior judgment signal.

---

## Stretch Goal (only if core is complete)

### Environmental-Chemical Panel (phthalates / phenols)
**CRITICAL weight-handling rule:** Environmental chemicals use a 1/3 subsample with different weights (e.g., `WTSB2YR` for phthalates). These MUST NOT be mixed with `WTMEC2YR`.

- Download `PHTHTE_J` (phthalates) and `EPH_J` (phenols/parabens)
- Build a separate analytic frame using ONLY the subsample with `WTSB2YR > 0`
- Re-run inference and ML on this subsample with `WTSB2YR` as the weight variable
- Report results separately — do not merge the two analytic frames into one model

---

## Deliverables Summary

| Deliverable | File Path | Produced By |
|-------------|-----------|------------|
| Analytic dataset | `/data/processed/analytic_table.parquet` | `02_build_dataset.py` |
| Weighted prevalence table | `/reports/weighted_prevalence_table.csv` | `03_eda_weighted.py` |
| EDA figures (4) | `/figures/fig_*.png` | `03_eda_weighted.py` |
| OR inference table | `/reports/inference_OR_table.csv` | `04_inference.py` |
| Forest plot | `/figures/fig_forest_plot_OR.png` | `04_inference.py` |
| ML metrics | `/reports/ml_metrics.json` | `05_predict_ml.py` |
| SHAP figures (4) | `/figures/fig_shap_*.png` | `05_predict_ml.py` |
| SHAP rankings | `/reports/shap_rankings.csv` | `05_predict_ml.py` |
| Technical report | `/reports/technical.md` | `06_reports.py` + human |
| Safety summary | `/reports/safety_summary.md` | `06_reports.py` + human |
| Methods/limitations | `/reports/methods_limitations.md` | `06_reports.py` + human |
| AI usage appendix | `/reports/ai_usage_appendix.md` | Human-written |
| Narrative notebook | `/notebooks/main.ipynb` | `main.ipynb` |
| README | `/README.md` | Human-written |
| One-command runner | `/run_all.sh` | Human-written |
