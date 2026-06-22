"""
LLM-assisted stakeholder report drafting using xAI Grok API.

Loads structured results from prior pipeline steps and drafts three reports:
  reports/technical.md          — R&D scientists
  reports/safety_summary.md     — Medical Safety / regulatory
  reports/methods_limitations.md — Regulatory / QA

Set XAI_API_KEY in environment before running.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "phase2_analysis"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "phase2_analysis"

GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Load pipeline outputs ─────────────────────────────────────────────────────

def load_inputs() -> dict:
    or_table   = pd.read_csv(REPORTS_DIR / "inference_OR_table.csv")
    or_sens    = pd.read_csv(REPORTS_DIR / "inference_OR_table_sensitivity.csv")
    shap       = pd.read_csv(REPORTS_DIR / "shap_rankings.csv")
    prevalence = pd.read_csv(REPORTS_DIR / "weighted_prevalence_table.csv")
    metrics    = json.loads((REPORTS_DIR / "ml_metrics.json").read_text())

    return {
        "or_table":   or_table,
        "or_sens":    or_sens,
        "shap":       shap,
        "prevalence": prevalence,
        "metrics":    metrics,
    }


def format_or_table(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        sig = "*" if r["p_value"] < 0.05 else ""
        rows.append(
            f"  {r['label']:<30} OR={r['OR']:.3f}  "
            f"95%CI [{r['CI_low']:.3f}–{r['CI_high']:.3f}]  "
            f"p={r['p_value']:.4f}{sig}"
        )
    return "\n".join(rows)


def format_shap(df: pd.DataFrame, n: int = 10) -> str:
    rows = []
    for _, r in df.head(n).iterrows():
        rows.append(f"  {r['label']:<30} mean|SHAP|={r['mean_abs_shap']:.4f}")
    return "\n".join(rows)


def format_prevalence(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


# ── Grok API call ─────────────────────────────────────────────────────────────

def call_grok(client: OpenAI, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


# ── Report drafters ───────────────────────────────────────────────────────────

def draft_technical(client: OpenAI, inputs: dict) -> str:
    system = (
        "You are a senior biostatistician writing a technical report for R&D scientists. "
        "Use precise statistical language. Do not overclaim causality. "
        "Reference exact numeric values from the data provided. "
        "Format as structured markdown with sections."
    )
    user = f"""Draft a technical report titled:
'NHANES 2017–2018: Population-Level Drivers and Individual-Risk Prediction for Elevated ALT'

Use exactly the results below — do not invent numbers.

## Survey-Weighted Logistic Regression — Odds Ratios (primary model, ALT > 40 U/L)
(* = p < 0.05; design-correct Taylor-series CIs; N ≈ 3,543)
{format_or_table(inputs['or_table'])}

## XGBoost ML Model — Test-Set Performance
AUC-ROC:     {inputs['metrics']['roc_auc']}
PR-AUC:      {inputs['metrics']['pr_auc']}
Sensitivity: {inputs['metrics']['sensitivity']}
Specificity: {inputs['metrics']['specificity']}
N test:      {inputs['metrics']['n_test']} ({inputs['metrics']['n_events_test']} events)
Features selected (RFECV): {inputs['metrics']['n_features_selected']} of {inputs['metrics']['n_features_total']}

## Top 10 Features by SHAP Importance
{format_shap(inputs['shap'], 10)}

## Weighted Prevalence Summary
{format_prevalence(inputs['prevalence'])}

Structure the report with these sections:
1. Background & Objective
2. Methods (survey design, model specification, ML pipeline)
3. Results (prevalence, OR table, ML metrics, SHAP rankings)
4. Discussion (clinical interpretation, key drivers, ML vs inference distinction)
5. Statistical Notes & Caveats
"""
    return call_grok(client, system, user)


def draft_safety_summary(client: OpenAI, inputs: dict) -> str:
    system = (
        "You are a medical safety writer preparing a plain-language summary for a "
        "Medical Safety / regulatory audience. Avoid jargon. Do not use causal language "
        "for ML-derived findings. Stick strictly to the numbers provided."
    )
    user = f"""Draft a safety summary titled:
'Elevated ALT in U.S. Adults: Key Risk Factors and Prevalence Findings (NHANES 2017–2018)'

Use only the results below:

Significant population-level risk factors (survey-weighted logistic regression):
{format_or_table(inputs['or_table'])}

Weighted prevalence by subgroup:
{format_prevalence(inputs['prevalence'])}

ML model performance (individual-level risk prediction):
AUC-ROC: {inputs['metrics']['roc_auc']}, Sensitivity: {inputs['metrics']['sensitivity']}, Specificity: {inputs['metrics']['specificity']}

Top predictors in ML model (SHAP):
{format_shap(inputs['shap'], 5)}

Write for a non-statistician audience. Use sections:
1. What We Studied
2. Who Is Most Affected (prevalence findings)
3. Key Risk Factors (from the statistical model — use plain language for ORs)
4. Individual Risk Screening (ML model — note this is a research tool, not a clinical test)
5. Implications for Safety Monitoring
"""
    return call_grok(client, system, user)


def draft_methods_limitations(client: OpenAI, inputs: dict) -> str:
    system = (
        "You are a regulatory affairs scientist writing a methods and limitations section "
        "for a QA/regulatory audience. Be precise about what the survey design does and does not allow. "
        "Be explicit about the boundaries of inference vs prediction."
    )
    user = f"""Draft a methods and limitations document titled:
'Methods, Statistical Design, and Limitations — NHANES 2017–2018 ALT Analysis'

Context:
- Data: NHANES 2017-2018, N={inputs['metrics']['n_test'] + 2834} after exclusions (hepatitis B/C negative, adults 18+, examined subsample)
- Outcome: ALT > 40 U/L (unisex threshold); sex-specific AASLD thresholds in sensitivity analysis
- Inference model: survey-weighted logistic regression with Taylor-series linearization (design-correct CIs)
  Predictors: age, sex, BMI, poverty-income ratio, diabetes, ever smoker, log(lead), log(cadmium), log(mercury)
- ML model: XGBoost with RFECV feature selection ({inputs['metrics']['n_features_selected']} features selected from {inputs['metrics']['n_features_total']})
  Test AUC-ROC: {inputs['metrics']['roc_auc']}; trained on unweighted data

Write sections:
1. Survey Design & Weighting (why WTMEC2YR is used; what Taylor-series linearization provides)
2. Outcome Definition (ALT threshold choice, sensitivity analysis rationale)
3. Exclusion Criteria & Sample Flow
4. Inference Model Specification & Assumptions
5. ML Model — Scope and Boundaries (trained for prediction, not causal inference; unweighted; full blood panel features)
6. Missing Data Approach
7. Key Limitations (cross-sectional design, single survey cycle, heavy-metal exposure measured at one time point)
8. What These Results Cannot Claim
"""
    return call_grok(client, system, user)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set. Export it before running:\n  export GROQ_API_KEY=your_key_here")
        raise SystemExit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    logger.info("Loading pipeline outputs ...")
    inputs = load_inputs()

    reports = {
        "technical.md":           ("Technical Report (R&D)",       draft_technical),
        "safety_summary.md":      ("Safety Summary (Medical/Reg)",  draft_safety_summary),
        "methods_limitations.md": ("Methods & Limitations (QA/Reg)", draft_methods_limitations),
    }

    for filename, (label, drafter) in reports.items():
        logger.info("Drafting %s ...", label)
        content = drafter(client, inputs)
        out_path = REPORTS_DIR / filename
        out_path.write_text(content, encoding="utf-8")
        logger.info("Saved %s", out_path)

    logger.info("\nAll three reports saved to %s", REPORTS_DIR)
    logger.info("Review each file before sharing — verify OR values match inference_OR_table.csv exactly.")


if __name__ == "__main__":
    main()
