# NHANES Liver Injury Risk Analysis — Kenvue Project

A reproducible, survey-design-correct pipeline for characterizing elevated ALT risk in the U.S. adult population using NHANES 2017–2018 and 2021–2022. Combines population-level inference (survey-weighted logistic regression) with individual-level prediction (XGBoost + SHAP) and an interactive AI agent layer.

---

## Key Results

| Metric | Value |
|--------|-------|
| Population prevalence of elevated ALT (>40 U/L) | 5.2% |
| ML model AUC-ROC | 0.973 |
| Sensitivity / Specificity | 0.926 / 0.922 |
| Analytic sample (N) | 3,543 |

**Significant independent predictors of elevated ALT:**
- Triglycerides (log): OR 1.77 (1.16–2.71) — strongest modifiable driver
- Waist circumference: OR 1.03 per cm (1.01–1.04)
- Age: OR 0.96 per year (0.94–0.98) — protective in adults (younger adults drive more NAFLD)

**Heavy metals (mercury, cadmium, lead) showed no independent association** after controlling for metabolic factors.

---

## Project Structure

```
├── src/                         # Analysis pipeline (run in order)
│   ├── 01_download.py           # Download NHANES 2017-2018 XPT files
│   ├── 02_build_dataset.py      # Merge modules, apply exclusions, feature engineering
│   ├── 03_eda_weighted.py       # Survey-weighted EDA figures
│   ├── 04_inference.py          # Survey-weighted logistic regression (Taylor-series CIs)
│   ├── 05_predict_ml.py         # XGBoost + SHAP prediction model
│   ├── 06_reports.py            # LLM-assisted report generation (Groq/Llama)
│   ├── 07_chat.py               # Streamlit AI chat interface
│   ├── 08_–17_*.py              # Extended hypotheses & COVID-era (2021-2022) comparison
│   └── agents/                  # Agentic AI layer (Anthropic SDK)
│       ├── tools.py             # 11 analysis tools incl. compute_individual_risk()
│       ├── qa_agent.py          # Q&A agent (personalized risk + stats lookup)
│       ├── analysis_agent.py    # Analysis agent (subgroup analysis)
│       └── orchestrator.py      # Routes queries between agents
│
├── data/
│   └── processed/               # Cleaned analytic tables (parquet + CSV)
│       ├── analytic_table.parquet       # 2017-2018, 3,543 rows, 225 cols
│       └── analytic_table_2021.parquet  # 2021-2022 for COVID-era comparison
│
├── reports/
│   ├── phase2_analysis/         # OR tables, SHAP rankings, ML metrics, methods docs
│   └── phase3_covid/            # 2017-2018 vs 2021-2022 comparison outputs
│
├── figures/
│   ├── phase1_eda/              # Weighted EDA figures
│   ├── phase2_analysis/         # Forest plots, SHAP plots, ROC curves
│   ├── phase3_covid/            # COVID-era comparison figures
│   └── slides_export/           # Figures used in the HM presentation
│
├── notebooks/
│   └── eda.ipynb                # Interactive EDA walkthrough
│
├── ROADMAP.md                   # Two-day execution plan with task checklist
└── PROJECT_DEEP_DIVE.md         # Full technical reference (design decisions, correctness rules)
```

---

## Quick Start

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys (needed for agent/report layers only)

```bash
export ANTHROPIC_API_KEY="your-key"   # for src/agents/ (AI chat)
export GROQ_API_KEY="your-key"        # for src/06_reports.py (LLM report drafting)
```

### 3. Run the pipeline

```bash
# Download raw NHANES data (~15 MB, cached in data/raw/)
python src/01_download.py

# Build analytic dataset
python src/02_build_dataset.py

# Weighted EDA
python src/03_eda_weighted.py

# Inference model (survey-weighted logistic regression)
python src/04_inference.py

# ML prediction model (XGBoost + SHAP)
python src/05_predict_ml.py

# Generate LLM-assisted reports
python src/06_reports.py
```

### 4. Launch the AI chat interface

```bash
streamlit run src/07_chat.py
# or CLI mode:
python src/07_chat.py --cli
```

---

## Methodology

### Why this differs from the published literature

Most NHANES ML papers (2021–2025) treat the complex probability sample as a simple random sample, which:
- Produces biased prevalence estimates (row proportions ≠ population proportions)
- Yields artificially narrow confidence intervals (ignores design effect)

This pipeline fixes both:
- **Inference layer** (`04_inference.py`): Survey-weighted logistic regression with Taylor-series linearization for design-correct SEs and CIs
- **ML layer** (`05_predict_ml.py`): XGBoost trained with survey weights; SHAP values for interpretability
- **Separation of concerns**: Inference answers "what drives population risk?"; ML answers "what is this individual's probability?"

### Exclusion criteria applied in `02_build_dataset.py`
- Age < 20 or missing
- Hepatitis B surface antigen positive
- Hepatitis C antibody positive
- Heavy alcohol use (>14 drinks/week men, >7 drinks/week women)
- Missing ALT or survey design variables

---

## COVID-Era Comparison (Phase 3)

Scripts `11_` through `17_` repeat the full pipeline on NHANES 2021–2022 data and compare:
- ALT prevalence: 2017-2018 vs. 2021-2022
- Model performance and feature importance shifts
- Wellness/lifestyle hypothesis testing across cycles

Figures in `figures/phase3_covid/` and reports in `reports/phase3_covid/`.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Step-by-step execution plan with time estimates |
| [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md) | Full technical reference — design decisions, correctness rules, deliverables map |
| [reports/phase2_analysis/methods_limitations.md](reports/phase2_analysis/methods_limitations.md) | Methods and limitations for the 2017-2018 analysis |
| [reports/phase2_analysis/technical.md](reports/phase2_analysis/technical.md) | Technical report with full model output |

---

## Data Source

All raw data is publicly available from CDC NHANES:
- 2017–2018: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
- 2021–2022: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2021

Raw `.XPT` files are **not stored in this repo** — run `src/01_download.py` to fetch them automatically.
