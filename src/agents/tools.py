"""
Tool functions and Anthropic tool schemas for the NHANES AI agent layer.

Q&A tools (fast, read-only):
  read_or_table, read_shap_rankings, read_ml_metrics,
  read_prevalence_table, read_report, compute_individual_risk

Analysis tools (compute, slower):
  query_analytic_table, run_subgroup_analysis,
  run_inference_script, run_ml_script, explain_risk_factors
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports" / "phase2_analysis"
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
SRC_DIR      = PROJECT_ROOT / "src"

# ── Module-level caches ────────────────────────────────────────────────────────
_df_cache: pd.DataFrame | None = None
_inference_mod = None


def _load_parquet() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet(DATA_DIR / "analytic_table.parquet")
    return _df_cache


def _load_inference_mod():
    global _inference_mod
    if _inference_mod is None:
        spec = importlib.util.spec_from_file_location("inference", SRC_DIR / "04_inference.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _inference_mod = mod
    return _inference_mod


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def read_or_table(sensitivity: bool = False) -> dict:
    """Return the survey-weighted logistic regression OR table."""
    fname = "inference_OR_table_sensitivity.csv" if sensitivity else "inference_OR_table.csv"
    path  = REPORTS_DIR / fname
    if not path.exists():
        return {"error": f"{fname} not found — run 04_inference.py first"}
    df = pd.read_csv(path)
    return {
        "model":       "sensitivity (sex-specific AASLD thresholds)" if sensitivity else "primary (unisex >40 U/L)",
        "n_sample":    3543,
        "population":  "U.S. civilian adults, NHANES 2017-2018",
        "method":      "Survey-weighted logistic regression, Taylor-series CIs",
        "results":     df.to_dict(orient="records"),
        "note":        "OR > 1 = higher odds of elevated ALT. All CIs are design-correct (not naive).",
    }


def read_shap_rankings(top_n: int = 10) -> dict:
    """Return top-N features by mean absolute SHAP value from the XGBoost model."""
    path = REPORTS_DIR / "shap_rankings.csv"
    if not path.exists():
        return {"error": "shap_rankings.csv not found — run 05_predict_ml.py first"}
    df = pd.read_csv(path).head(top_n)
    return {
        "model":   "XGBoost (individual prediction, not population-level inference)",
        "metric":  "mean absolute SHAP value — average contribution to predicted log-odds",
        "top_features": df.to_dict(orient="records"),
        "warning": "AST/GGT appear top because they are liver enzymes correlated with ALT. "
                   "For the 9-predictor epidemiologic model (no lab values), see read_or_table().",
    }


def read_ml_metrics() -> dict:
    """Return XGBoost model performance metrics."""
    path = REPORTS_DIR / "ml_metrics.json"
    if not path.exists():
        return {"error": "ml_metrics.json not found — run 05_predict_ml.py first"}
    with open(path) as f:
        metrics = json.load(f)
    metrics["interpretation"] = (
        "AUC-ROC of 0.973 means the model ranks a random positive above a random negative "
        "97.3% of the time. Sensitivity 0.926 = catches 92.6% of true elevated-ALT cases."
    )
    return metrics


def read_prevalence_table(group: str | None = None) -> dict:
    """Return weighted prevalence of elevated ALT by demographic group."""
    path = REPORTS_DIR / "weighted_prevalence_table.csv"
    if not path.exists():
        return {"error": "weighted_prevalence_table.csv not found — run 03_eda_weighted.py first"}
    df = pd.read_csv(path)
    if group:
        mask = df["Group"].str.lower().str.contains(group.lower(), na=False)
        df   = df[mask]
        if df.empty:
            available = df["Group"].unique().tolist()
            return {"error": f"No rows matching '{group}'. Available groups: {available}"}
    return {
        "outcome":     "Elevated ALT (>40 U/L, unisex threshold)",
        "population":  "U.S. civilian adults, NHANES 2017-2018, survey-weighted",
        "rows":        df.to_dict(orient="records"),
    }


def read_report(report_name: str) -> dict:
    """Return the full text of a stakeholder report.

    report_name: one of 'technical', 'safety_summary', 'methods_limitations'
    """
    valid = {"technical", "safety_summary", "methods_limitations"}
    name  = report_name.lower().replace(" ", "_").replace("-", "_")
    if name not in valid:
        return {"error": f"Unknown report '{report_name}'. Choose from: {sorted(valid)}"}
    path = REPORTS_DIR / f"{name}.md"
    if not path.exists():
        return {"error": f"{name}.md not found — run 06_reports.py first"}
    return {
        "report":  name,
        "content": path.read_text(encoding="utf-8"),
    }


def compute_individual_risk(
    age: float,
    sex_male: int,
    diabetes: int,
    waist_cm: float | None = None,
    triglycerides_mg_dl: float | None = None,
    ever_smoker: int | None = None,
    poverty_income_ratio: float | None = None,
    blood_lead_ug_dl: float | None = None,
    blood_cadmium_ug_l: float | None = None,
    blood_mercury_ug_l: float | None = None,
) -> dict:
    """Estimate individual ALT elevation risk from the 10-predictor survey-weighted model.

    Required: age, sex_male (1/0), diabetes (1/0).
    Optional: waist_cm, triglycerides_mg_dl, ever_smoker, poverty_income_ratio, metals.
    Missing values are imputed from survey-weighted population means.
    """
    or_path = REPORTS_DIR / "inference_OR_table.csv"
    if not or_path.exists():
        return {"error": "OR table not found — run 04_inference.py first"}

    or_df = pd.read_csv(or_path)
    betas = dict(zip(or_df["variable"], or_df["beta"]))

    df = _load_parquet()

    # Population-weighted means for calibration + imputation of missing values
    def wmean(col: str) -> float:
        mask = df[col].notna() & df["WTMEC2YR"].notna()
        return float((df.loc[mask, col] * df.loc[mask, "WTMEC2YR"]).sum() / df.loc[mask, "WTMEC2YR"].sum())

    # Derive log_triglycerides column if not already in parquet
    if "log_triglycerides" not in df.columns and "LBXSTR" in df.columns:
        import numpy as _np
        df = df.copy()
        df["log_triglycerides"] = _np.log(df["LBXSTR"].clip(lower=1))

    pop_means: dict[str, float] = {
        "RIDAGEYR":          wmean("RIDAGEYR"),
        "sex_male":          wmean("sex_male") if "sex_male" in df.columns else 0.49,
        "INDFMPIR":          wmean("INDFMPIR"),
        "BMXWAIST":          wmean("BMXWAIST"),
        "diabetes":          wmean("diabetes") if "diabetes" in df.columns else 0.12,
        "ever_smoker":       wmean("ever_smoker") if "ever_smoker" in df.columns else 0.43,
        "log_triglycerides": wmean("log_triglycerides") if "log_triglycerides" in df.columns else math.log(120),
        "log_lead":          wmean("log_lead") if "log_lead" in df.columns else math.log(1.15),
        "log_cadmium":       wmean("log_cadmium") if "log_cadmium" in df.columns else math.log(0.25),
        "log_mercury":       wmean("log_mercury") if "log_mercury" in df.columns else math.log(0.8),
    }

    # Calibrate intercept: "average person" → population prevalence (5.2%)
    pop_prevalence = 0.052
    pop_logit = math.log(pop_prevalence / (1 - pop_prevalence))
    pop_linear = sum(betas[var] * pop_means[var] for var in betas)
    intercept  = pop_logit - pop_linear

    # Build individual values; impute missing with pop means
    waist       = float(waist_cm)                if waist_cm             is not None else pop_means["BMXWAIST"]
    log_trig    = math.log(max(triglycerides_mg_dl, 1)) if triglycerides_mg_dl is not None else pop_means["log_triglycerides"]
    log_lead    = math.log(blood_lead_ug_dl)     if blood_lead_ug_dl     is not None else pop_means["log_lead"]
    log_cadmium = math.log(blood_cadmium_ug_l)   if blood_cadmium_ug_l  is not None else pop_means["log_cadmium"]
    log_mercury = math.log(blood_mercury_ug_l)   if blood_mercury_ug_l  is not None else pop_means["log_mercury"]
    pir         = poverty_income_ratio            if poverty_income_ratio is not None else pop_means["INDFMPIR"]
    smoker      = float(ever_smoker)              if ever_smoker          is not None else pop_means["ever_smoker"]

    person = {
        "RIDAGEYR":          float(age),
        "sex_male":          float(sex_male),
        "INDFMPIR":          float(pir),
        "BMXWAIST":          waist,
        "diabetes":          float(diabetes),
        "ever_smoker":       smoker,
        "log_triglycerides": log_trig,
        "log_lead":          log_lead,
        "log_cadmium":       log_cadmium,
        "log_mercury":       log_mercury,
    }

    linear_pred = intercept + sum(betas[var] * person[var] for var in betas)
    prob        = 1.0 / (1.0 + math.exp(-linear_pred))

    pop_odds    = pop_prevalence / (1 - pop_prevalence)
    person_odds = prob / (1 - prob)
    relative_odds = person_odds / pop_odds

    # Top risk drivers: contribution of each predictor relative to average person
    drivers = []
    for var, beta in betas.items():
        contribution = beta * (person[var] - pop_means[var])
        label = or_df.loc[or_df["variable"] == var, "label"].values[0] if var in or_df["variable"].values else var
        if abs(contribution) > 0.001:
            drivers.append({"label": label, "contribution_to_log_odds": round(contribution, 4)})
    drivers.sort(key=lambda x: abs(x["contribution_to_log_odds"]), reverse=True)

    if relative_odds >= 2.0:
        interpretation = "substantially higher than average"
    elif relative_odds >= 1.25:
        interpretation = "higher than average"
    elif relative_odds <= 0.5:
        interpretation = "substantially lower than average"
    elif relative_odds <= 0.8:
        interpretation = "lower than average"
    else:
        interpretation = "near the population average"

    return {
        "estimated_risk_pct":        round(prob * 100, 1),
        "population_average_pct":    5.2,
        "relative_odds_vs_average":  round(relative_odds, 2),
        "risk_interpretation":       interpretation,
        "top_risk_drivers":          drivers[:5],
        "inputs_used": {
            "age": age, "sex_male": sex_male, "waist_cm": waist_cm, "diabetes": diabetes,
            "triglycerides_mg_dl": triglycerides_mg_dl,
            "waist_imputed": waist_cm is None,
            "triglycerides_imputed": triglycerides_mg_dl is None,
            "metals_imputed": blood_lead_ug_dl is None,
        },
        "disclaimer": (
            "Statistical estimate from NHANES 2017-2018 population survey. "
            "Not a clinical diagnostic test. Consult a physician for medical advice."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_FILTER_RE = re.compile(r'^[\w\s\d\.\'"=!<>&|()\[\],+\-\.]+$')


def query_analytic_table(
    filter_expression: str,
    columns: list[str],
    use_weights: bool = True,
) -> dict:
    """Run a filtered aggregate query on the analytic table parquet.

    Returns weighted prevalence and counts — never row-level data.
    filter_expression: pandas query string, e.g. "RIDRETH3 in [1,2] and RIAGENDR==1 and RIDAGEYR>40"
    columns: outcome columns to summarise, e.g. ["ALT_elevated_40"]
    """
    if not _SAFE_FILTER_RE.match(filter_expression):
        return {"error": "Invalid filter expression — only alphanumeric and basic operators allowed"}

    df = _load_parquet()
    try:
        sub = df.query(filter_expression)
    except Exception as e:
        return {"error": f"Query failed: {e}"}

    if len(sub) < 10:
        return {"error": f"Subgroup too small (n={len(sub)}). Cannot produce reliable estimates."}

    results: dict[str, Any] = {"n_unweighted": len(sub), "filter": filter_expression}

    for col in columns:
        if col not in sub.columns:
            results[col] = f"Column '{col}' not found"
            continue
        valid = sub[col].notna()
        if use_weights and "WTMEC2YR" in sub.columns:
            w    = sub.loc[valid, "WTMEC2YR"]
            vals = sub.loc[valid, col]
            wmean_val = float((vals * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
            results[col] = {
                "weighted_mean_pct":   round(wmean_val * 100, 2) if sub[col].between(0, 1).all() else round(wmean_val, 4),
                "n_valid":             int(valid.sum()),
                "note": "survey-weighted" if use_weights else "unweighted",
            }
        else:
            results[col] = {
                "unweighted_mean": round(float(sub.loc[valid, col].mean()), 4),
                "n_valid":         int(valid.sum()),
            }

    if len(sub) < 500:
        results["warning"] = f"Small subgroup (n={len(sub)}). CIs will be wide; interpret estimates cautiously."

    return results


def run_subgroup_analysis(
    filter_expression: str,
    label: str,
    outcome: str = "ALT_elevated_40",
) -> dict:
    """Fit survey-weighted logistic regression in a subgroup of the analytic table.

    filter_expression: pandas query string defining the subgroup
    label: human label for this subgroup (used in output)
    outcome: binary outcome column (default: ALT_elevated_40)
    """
    if not _SAFE_FILTER_RE.match(filter_expression):
        return {"error": "Invalid filter expression"}

    df = _load_parquet()
    try:
        sub = df.query(filter_expression).copy()
    except Exception as e:
        return {"error": f"Query failed: {e}"}

    if len(sub) < 200:
        return {"error": f"Subgroup too small for logistic regression (n={len(sub)}, need ≥200)."}

    mod = _load_inference_mod()

    try:
        sub = mod.prepare_features(sub)
        or_table = mod.run_model(sub, outcome)
    except Exception as e:
        return {"error": f"Model failed: {e}"}

    return {
        "subgroup":    label,
        "filter":      filter_expression,
        "n_subgroup":  len(sub),
        "outcome":     outcome,
        "or_table":    or_table.to_dict(orient="records"),
        "note":        "Survey-weighted ORs within this subgroup only. CIs may be wide for small n.",
    }


def run_inference_script(confirm: bool = False) -> dict:
    """Re-run 04_inference.py to regenerate OR tables and forest plot."""
    if not confirm:
        return {
            "action_required": "confirm",
            "message": "This will re-run 04_inference.py (~30–60 sec). Set confirm=true to proceed.",
        }
    script = SRC_DIR / "04_inference.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    return {
        "returncode": result.returncode,
        "stdout":     result.stdout[-3000:] if result.stdout else "",
        "stderr":     result.stderr[-2000:] if result.stderr else "",
        "status":     "success" if result.returncode == 0 else "error",
    }


def run_ml_script(confirm: bool = False) -> dict:
    """Re-run 05_predict_ml.py to regenerate the XGBoost model and SHAP plots."""
    if not confirm:
        return {
            "action_required": "confirm",
            "message": "This will re-run 05_predict_ml.py (3–10 min). Set confirm=true to proceed.",
        }
    script = SRC_DIR / "05_predict_ml.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    return {
        "returncode": result.returncode,
        "stdout":     result.stdout[-3000:] if result.stdout else "",
        "stderr":     result.stderr[-2000:] if result.stderr else "",
        "status":     "success" if result.returncode == 0 else "error",
    }


def explain_risk_factors(focus_factor: str, audience: str = "general") -> dict:
    """Pre-load OR + SHAP context for a specific risk factor to seed agent explanation.

    focus_factor: e.g. 'diabetes', 'bmi', 'sex', 'lead', 'age'
    audience: 'general' | 'clinical' | 'regulatory'
    """
    or_df   = pd.read_csv(REPORTS_DIR / "inference_OR_table.csv") if (REPORTS_DIR / "inference_OR_table.csv").exists() else None
    shap_df = pd.read_csv(REPORTS_DIR / "shap_rankings.csv") if (REPORTS_DIR / "shap_rankings.csv").exists() else None

    factor  = focus_factor.lower()
    context: dict[str, Any] = {"focus_factor": focus_factor, "audience": audience}

    if or_df is not None:
        mask = or_df["label"].str.lower().str.contains(factor, na=False) | \
               or_df["variable"].str.lower().str.contains(factor, na=False)
        context["or_evidence"] = or_df[mask].to_dict(orient="records") if mask.any() else []

    if shap_df is not None:
        mask2 = shap_df["label"].str.lower().str.contains(factor, na=False) if "label" in shap_df.columns else pd.Series(False, index=shap_df.index)
        context["shap_evidence"] = shap_df[mask2].to_dict(orient="records") if mask2.any() else []

    context["interpretation_guidance"] = {
        "or_meaning": "Population-level association in U.S. adults (cross-sectional, not causal)",
        "shap_meaning": "Individual prediction importance in XGBoost model",
        "cross_sectional_caveat": "Cannot establish causation from NHANES design",
    }
    return context


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DISPATCH + SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_FN_MAP: dict[str, Any] = {
    "read_or_table":           read_or_table,
    "read_shap_rankings":      read_shap_rankings,
    "read_ml_metrics":         read_ml_metrics,
    "read_prevalence_table":   read_prevalence_table,
    "read_report":             read_report,
    "compute_individual_risk": compute_individual_risk,
    "query_analytic_table":    query_analytic_table,
    "run_subgroup_analysis":   run_subgroup_analysis,
    "run_inference_script":    run_inference_script,
    "run_ml_script":           run_ml_script,
    "explain_risk_factors":    explain_risk_factors,
}

QA_TOOL_NAMES       = ["read_or_table", "read_shap_rankings", "read_ml_metrics",
                        "read_prevalence_table", "read_report", "compute_individual_risk"]
ANALYSIS_TOOL_NAMES = list(_TOOL_FN_MAP.keys())

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "read_or_table",
        "description": "Return the survey-weighted logistic regression OR table (10 predictors: age, sex, PIR, waist circumference, diabetes, smoking, triglycerides, lead, cadmium, mercury). Use for odds ratio lookups, hypothesis test results, or p-values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sensitivity": {
                    "type": "boolean",
                    "description": "If true, return sensitivity model (sex-specific AASLD thresholds). Default false (unisex >40 U/L)."
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_shap_rankings",
        "description": "Return top-N features by mean absolute SHAP value from the XGBoost individual prediction model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top features to return. Default 10."
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_ml_metrics",
        "description": "Return XGBoost model performance: AUC-ROC, PR-AUC, sensitivity, specificity, confusion matrix.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_prevalence_table",
        "description": "Return survey-weighted prevalence of elevated ALT by demographic group (sex, age, race, BMI category, diabetes).",
        "input_schema": {
            "type": "object",
            "properties": {
                "group": {
                    "type": "string",
                    "description": "Filter by group name (e.g. 'Sex', 'Age', 'Race', 'BMI', 'Diabetes'). Omit for all rows."
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_report",
        "description": "Return the full text of a stakeholder report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "report_name": {
                    "type": "string",
                    "enum": ["technical", "safety_summary", "methods_limitations"],
                    "description": "Which report to retrieve."
                }
            },
            "required": ["report_name"],
        },
    },
    {
        "name": "compute_individual_risk",
        "description": (
            "Estimate the probability of elevated ALT for a specific person using the "
            "10-predictor survey-weighted logistic regression. Returns estimated risk %, "
            "comparison to population average (5.2%), relative odds, and top risk drivers. "
            "ALWAYS call this when the user describes a person's profile. "
            "Key new predictors: waist_cm (replaces BMI) and triglycerides_mg_dl."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age":                    {"type": "number", "description": "Age in years (18–80)"},
                "sex_male":               {"type": "integer", "enum": [0, 1], "description": "1=male, 0=female"},
                "diabetes":               {"type": "integer", "enum": [0, 1], "description": "1=diagnosed diabetes, 0=no"},
                "waist_cm":               {"type": "number", "description": "Waist circumference in cm. Imputed from pop mean if omitted."},
                "triglycerides_mg_dl":    {"type": "number", "description": "Triglycerides in mg/dL. Imputed from pop mean if omitted."},
                "ever_smoker":            {"type": "integer", "enum": [0, 1], "description": "1=ever smoked 100+ cigarettes, 0=never"},
                "poverty_income_ratio":   {"type": "number", "description": "Poverty-income ratio (0–5). Omit if unknown."},
                "blood_lead_ug_dl":       {"type": "number", "description": "Blood lead µg/dL. Omit if unknown (imputed from population mean)."},
                "blood_cadmium_ug_l":     {"type": "number", "description": "Blood cadmium µg/L. Omit if unknown."},
                "blood_mercury_ug_l":     {"type": "number", "description": "Blood mercury µg/L. Omit if unknown."},
            },
            "required": ["age", "sex_male", "diabetes"],
        },
    },
    {
        "name": "query_analytic_table",
        "description": (
            "Run a filtered aggregate query on the NHANES analytic table parquet. "
            "Returns survey-weighted prevalence for a custom subgroup. "
            "Use for subgroups NOT in the prevalence table (e.g. Hispanic males over 40). "
            "Key columns: RIAGENDR (1=male, 2=female), RIDRETH3 (1=Mex.Am, 2=Oth.Hisp, 3=NHWhite, 4=NHBlack, 6=NHAsian), "
            "RIDAGEYR, BMXBMI, diabetes (1/0), sex_male (1/0), ALT_elevated_40 (1/0), WTMEC2YR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_expression": {
                    "type": "string",
                    "description": "Pandas query string, e.g. 'RIDRETH3 in [1,2] and RIAGENDR==1 and RIDAGEYR>40'"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Outcome columns to summarise, e.g. ['ALT_elevated_40']"
                },
                "use_weights": {
                    "type": "boolean",
                    "description": "Use survey weights (WTMEC2YR) for weighted prevalence. Default true."
                },
            },
            "required": ["filter_expression", "columns"],
        },
    },
    {
        "name": "run_subgroup_analysis",
        "description": (
            "Fit a full survey-weighted logistic regression in a subgroup of the analytic table. "
            "Use when asked to 'run the analysis for diabetic women' or similar. Takes ~5–15 seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_expression": {
                    "type": "string",
                    "description": "Pandas query string defining the subgroup, e.g. 'diabetes==1 and sex_male==0'"
                },
                "label": {
                    "type": "string",
                    "description": "Human-readable label, e.g. 'Diabetic women'"
                },
                "outcome": {
                    "type": "string",
                    "description": "Binary outcome column. Default 'ALT_elevated_40'."
                },
            },
            "required": ["filter_expression", "label"],
        },
    },
    {
        "name": "run_inference_script",
        "description": "Re-run 04_inference.py to regenerate OR tables and forest plot (~30–60 sec). Requires confirm=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "Set to true to actually run the script."}
            },
            "required": [],
        },
    },
    {
        "name": "run_ml_script",
        "description": "Re-run 05_predict_ml.py to regenerate the XGBoost model and SHAP plots (3–10 min). Requires confirm=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "Set to true to actually run the script."}
            },
            "required": [],
        },
    },
    {
        "name": "explain_risk_factors",
        "description": "Pre-load OR and SHAP evidence for a specific risk factor to support a detailed explanation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus_factor": {
                    "type": "string",
                    "description": "Factor to explain, e.g. 'diabetes', 'bmi', 'sex', 'lead', 'age'"
                },
                "audience": {
                    "type": "string",
                    "enum": ["general", "clinical", "regulatory"],
                    "description": "Target audience for the explanation. Default 'general'."
                },
            },
            "required": ["focus_factor"],
        },
    },
]


def get_tool_schemas(names: list[str]) -> list[dict]:
    """Return Anthropic tool schemas for the specified tool names."""
    name_set = set(names)
    return [s for s in TOOL_SCHEMAS if s["name"] in name_set]


def execute_tool(name: str, args: dict) -> Any:
    """Dispatch a tool call by name."""
    fn = _TOOL_FN_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: '{name}'"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"Tool '{name}' raised {type(e).__name__}: {e}"}
