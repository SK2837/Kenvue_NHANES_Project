"""
Centralised path constants for the NHANES Kenvue project.

Import this in any script to get consistent, phase-organised paths:

    from paths import PROCESSED_DIR, FIG_ANALYSIS, RPT_ANALYSIS

Structure
---------
data/
  raw/            ← NHANES 2017-2018 XPT files (_J suffix)
  raw_2021/       ← NHANES 2021-2023 XPT files (_L suffix)
  processed/      ← Parquet analytic tables + codebooks

figures/
  phase1_eda/     ← EDA charts (distributions, correlations, race)
  phase2_analysis/← Analysis figures (forest plots, SHAP, ROC)
  phase3_covid/   ← Comparison figures (COVID impact, model comparison)

reports/
  phase2_analysis/← OR tables, ML metrics, stakeholder reports
  phase3_covid/   ← Comparison CSVs, wellness hypothesis tables
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_2017_DIR  = DATA_DIR / "raw"
RAW_2021_DIR  = DATA_DIR / "raw_2021"

# ── Figures (by phase) ────────────────────────────────────────────────────────
FIGURES_ROOT = ROOT / "figures"
FIG_EDA      = FIGURES_ROOT / "phase1_eda"       # 03_eda_weighted.py
FIG_ANALYSIS = FIGURES_ROOT / "phase2_analysis"   # 04, 05, 08, 10
FIG_COVID    = FIGURES_ROOT / "phase3_covid"      # 13-17

# ── Reports (by phase) ────────────────────────────────────────────────────────
REPORTS_ROOT = ROOT / "reports"
RPT_ANALYSIS = REPORTS_ROOT / "phase2_analysis"   # 03-09
RPT_COVID    = REPORTS_ROOT / "phase3_covid"      # 13-17
