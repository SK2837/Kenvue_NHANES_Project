"""
Phase 3 Comparison Figures — Pre/Post COVID Summary Dashboard
=============================================================
Assembles a multi-panel comparison figure combining the key Phase 3
findings into a single presentation-ready dashboard, plus individual
high-resolution figures for the Streamlit app.

Panels:
  A. ALT & wellness prevalence shift (bar chart)
  B. Model OR comparison: significant predictors
  C. SHAP importance shift (top 10)
  D. ML performance comparison (ROC AUC)

Outputs:
  figures/fig_phase3_dashboard.png    — 4-panel summary
  figures/fig_covid_impact_summary.png — narrative COVID impact figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "phase3_covid"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "phase3_covid"

COLOR_17 = "#4C72B0"
COLOR_21 = "#DD8452"


def load_data() -> dict:
    data = {}

    # Prevalence comparison
    prev = pd.read_csv(REPORTS_DIR / "prevalence_comparison.csv")
    data["prev"] = prev

    # Model comparison
    mc = pd.read_csv(REPORTS_DIR / "model_comparison_combined.csv")
    data["model"] = mc

    # SHAP comparison
    shap = pd.read_csv(REPORTS_DIR / "shap_comparison.csv")
    data["shap"] = shap

    # Wellness hypothesis
    wh = pd.read_csv(REPORTS_DIR / "wellness_hypothesis_combined.csv")
    data["wellness"] = wh

    return data


# ── Panel helpers ──────────────────────────────────────────────────────────────

def panel_prevalence(ax: plt.Axes, prev: pd.DataFrame) -> None:
    """Panel A: Key prevalence shifts."""
    focus_labels = [
        "ALT > 40 U/L (all)",
        "Depression (PHQ-9 ≥10)",
        "High sedentary (≥8h/day)",
        "Short sleep (<7h)",
        "Diabetes",
        "Obesity (BMI≥30)",
    ]
    p17 = prev[prev["cohort"] == "2017-2018"].set_index("label")["prev_pct"]
    p21 = prev[prev["cohort"] == "2021-2023"].set_index("label")["prev_pct"]

    labels = [l for l in focus_labels if l in p17.index and l in p21.index]
    v17 = [p17[l] for l in labels]
    v21 = [p21[l] for l in labels]

    y = np.arange(len(labels))
    w = 0.38
    ax.barh(y + w/2, v17, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    ax.barh(y - w/2, v21, w, color=COLOR_21, alpha=0.85, label="2021-2023")

    for i, (v_17, v_21) in enumerate(zip(v17, v21)):
        delta = v_21 - v_17
        sign = "+" if delta >= 0 else ""
        col = "#B22222" if delta > 1 else ("#006400" if delta < -1 else "gray")
        ax.text(max(v_17, v_21) + 0.5, i, f"{sign}{delta:.1f}pp",
                va="center", fontsize=7.5, color=col, fontweight="bold" if abs(delta) > 2 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Weighted prevalence (%)", fontsize=8)
    ax.set_title("A. Key Prevalence Shifts", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.invert_yaxis()


def panel_model_ors(ax: plt.Axes, mc: pd.DataFrame) -> None:
    """Panel B: Significant predictor OR comparison."""
    # Show predictors that are significant in at least one cohort
    sig = mc[(mc["p_2017"] < 0.1) | (mc["p_2021"] < 0.1)].copy()
    sig = sig.sort_values("OR_2017", ascending=True)

    labels = sig["label"].tolist()
    or17   = sig["OR_2017"].tolist()
    or21   = sig["OR_2021"].tolist()
    lo17   = sig["CI_low_2017"].tolist()
    hi17   = sig["CI_high_2017"].tolist()
    lo21   = sig["CI_low_2021"].tolist()
    hi21   = sig["CI_high_2021"].tolist()
    p17    = sig["p_2017"].tolist()
    p21    = sig["p_2021"].tolist()

    y = np.arange(len(labels))
    off = 0.18

    for i in range(len(labels)):
        for (or_v, lo, hi, pv, ypos, col) in [
            (or17[i], lo17[i], hi17[i], p17[i], y[i]+off, COLOR_17),
            (or21[i], lo21[i], hi21[i], p21[i], y[i]-off, COLOR_21),
        ]:
            if np.isnan(or_v):
                continue
            marker = "D" if pv < 0.05 else "o"
            ax.errorbar(or_v, ypos,
                        xerr=[[or_v - lo], [hi - or_v]],
                        fmt=marker, color=col, markersize=6, capsize=3,
                        linewidth=1.5, elinewidth=1.5, alpha=0.9)
            if pv < 0.05:
                ax.text(hi + 0.04, ypos, "*", ha="left", va="center",
                        fontsize=10, color=col, fontweight="bold")

    ax.axvline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Adjusted OR (95% CI)", fontsize=8)
    ax.set_title("B. Epidemiologic Model OR Comparison (* p<0.05)", fontsize=9, fontweight="bold")
    ax.set_xlim(left=0)

    p17_patch = mpatches.Patch(color=COLOR_17, label="2017-2018")
    p21_patch = mpatches.Patch(color=COLOR_21, label="2021-2023")
    ax.legend(handles=[p17_patch, p21_patch], fontsize=7.5)


def panel_shap(ax: plt.Axes, shap: pd.DataFrame) -> None:
    """Panel C: SHAP importance comparison (top 10)."""
    top = shap.head(10)
    labels = [str(l)[:20] for l in top["label"].tolist()]
    s17    = top["shap_2017"].tolist()
    s21    = top["shap_2021"].tolist()

    y = np.arange(len(labels))
    w = 0.38
    ax.barh(y + w/2, s17, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    ax.barh(y - w/2, s21, w, color=COLOR_21, alpha=0.85, label="2021-2023")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=8)
    ax.set_title("C. Top 10 SHAP Features (XGBoost)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.invert_yaxis()


def panel_ml_metrics(ax: plt.Axes) -> None:
    """Panel D: ML performance comparison."""
    import json
    metrics_17_path = Path(__file__).parent.parent / "reports" / "phase2_analysis" / "ml_metrics.json"
    metrics_21_path = REPORTS_DIR / "ml_metrics_2021.json"

    if not metrics_17_path.exists() or not metrics_21_path.exists():
        ax.text(0.5, 0.5, "ML metrics not found\nRun 05 and 16 scripts first",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        return

    with open(metrics_17_path) as f:
        m17 = json.load(f)
    with open(metrics_21_path) as f:
        m21 = json.load(f)

    metrics = ["AUC-ROC", "PR-AUC"]
    v17 = [m17.get("roc_auc", 0), m17.get("pr_auc", 0)]
    v21 = [m21.get("roc_auc", 0), m21.get("pr_auc", 0)]

    x = np.arange(len(metrics))
    w = 0.3
    b1 = ax.bar(x - w/2, v17, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    b2 = ax.bar(x + w/2, v21, w, color=COLOR_21, alpha=0.85, label="2021-2023")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("D. ML Model Performance", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", lw=0.5, alpha=0.4)


# ── Dashboard figure ──────────────────────────────────────────────────────────

def make_dashboard(data: dict, path: Path) -> None:
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_prevalence(ax_a, data["prev"])
    panel_model_ors(ax_b, data["model"])
    panel_shap(ax_c, data["shap"])
    panel_ml_metrics(ax_d)

    fig.suptitle(
        "Phase 3: NHANES 2017-2018 vs 2021-2023 Cohort Comparison — ALT Analysis\n"
        "Survey-weighted | 10-predictor logistic model | XGBoost SHAP\n"
        "(Descriptive cross-cohort comparison; differences reflect cohort, weighting, and cycle-length variation)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── COVID impact narrative figure ──────────────────────────────────────────────

def make_covid_impact_figure(data: dict, path: Path) -> None:
    """Cohort-comparison figure: what differed between NHANES 2017-2018 and 2021-2023."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "NHANES 2017-2018 vs 2021-2023: Liver Health Cohort Comparison",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # 1. Population wellness changes
    ax = axes[0]
    items = [
        ("ALT > 40 U/L",      8.15, 7.36),
        ("Depression\n(PHQ-9≥10)", 8.16, 11.76),
        ("High sedentary\n(≥8h/day)", 30.24, 33.63),
        ("Short sleep\n(<7h)",  24.99, 20.26),
    ]
    labels = [i[0] for i in items]
    v17    = [i[1] for i in items]
    v21    = [i[2] for i in items]
    x = np.arange(len(items))
    w = 0.35
    ax.bar(x - w/2, v17, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    ax.bar(x + w/2, v21, w, color=COLOR_21, alpha=0.85, label="2021-2023")
    for i, (a, b) in enumerate(zip(v17, v21)):
        delta = b - a
        sign = "+" if delta >= 0 else ""
        col = "#B22222" if delta > 1 else ("#006400" if delta < -1 else "gray")
        ax.text(i, max(a, b) + 0.8, f"{sign}{delta:.1f}pp",
                ha="center", fontsize=8.5, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Weighted prevalence (%)", fontsize=9)
    ax.set_title("Population Health Shifts", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 2. Predictors that changed significance
    ax = axes[1]
    # Show OR for the predictors that changed meaningfully
    mc = data["model"]
    preds_of_interest = ["Male sex", "Diabetes", "Triglycerides (log)", "Waist circumference (cm)", "Age (years)"]
    mc_sub = mc[mc["label"].isin(preds_of_interest)].set_index("label")

    or_17  = [mc_sub.loc[l, "OR_2017"]   if l in mc_sub.index else np.nan for l in preds_of_interest]
    or_21  = [mc_sub.loc[l, "OR_2021"]   if l in mc_sub.index else np.nan for l in preds_of_interest]
    p_17   = [mc_sub.loc[l, "p_2017"]    if l in mc_sub.index else 1     for l in preds_of_interest]
    p_21   = [mc_sub.loc[l, "p_2021"]    if l in mc_sub.index else 1     for l in preds_of_interest]
    short_labels = ["Male sex", "Diabetes", "Triglycerides\n(log)", "Waist\n(cm)", "Age"]

    x = np.arange(len(preds_of_interest))
    w = 0.35
    bars17 = ax.bar(x - w/2, or_17, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    bars21 = ax.bar(x + w/2, or_21, w, color=COLOR_21, alpha=0.85, label="2021-2023")

    for i, (b, pv) in enumerate(zip(bars17, p_17)):
        if pv < 0.05:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    "*", ha="center", fontsize=12, color=COLOR_17, fontweight="bold")
    for i, (b, pv) in enumerate(zip(bars21, p_21)):
        if pv < 0.05:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    "*", ha="center", fontsize=12, color=COLOR_21, fontweight="bold")

    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel("Adjusted Odds Ratio", fontsize=9)
    ax.set_title("Key OR Shifts (* p<0.05)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 3. Wellness variable hypothesis test (adjusted ORs)
    ax = axes[2]
    wh = data["wellness"]
    w_labels = wh["label"].tolist()
    or_17_w  = wh["OR_adj_2017"].tolist()
    or_21_w  = wh["OR_adj_2021"].tolist()
    p_17_w   = wh["p_adj_2017"].tolist()
    p_21_w   = wh["p_adj_2021"].tolist()

    x = np.arange(len(w_labels))
    w = 0.35
    b17 = ax.bar(x - w/2, or_17_w, w, color=COLOR_17, alpha=0.85, label="2017-2018")
    b21 = ax.bar(x + w/2, or_21_w, w, color=COLOR_21, alpha=0.85, label="2021-2023")

    for bars, pvals in [(b17, p_17_w), (b21, p_21_w)]:
        for bar, pv in zip(bars, pvals):
            if pv < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        "*", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in w_labels], fontsize=8)
    ax.set_ylabel("Adjusted OR for ALT>40", fontsize=9)
    ax.set_title(
        "Wellness Variables as ALT Predictors\n(adjusted for 10-predictor model; NS = not significant)",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8)

    # Annotation box
    fig.text(
        0.5, -0.04,
        "Lifestyle variables (depression +3.6pp, sedentary +3.4pp) increased in prevalence between cohorts "
        "but were not significant ALT predictors in either cohort after metabolic adjustment.\n"
        "The male sex association was stronger in the 2021-2023 cohort (OR 2.03→2.93) "
        "while the diabetes association was attenuated — cohort differences that warrant further investigation.\n"
        "Note: 2021-2023 uses a 3-year survey cycle with updated design weights; "
        "this comparison is descriptive and does not establish a causal COVID effect.",
        ha="center", va="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#E5A800", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for req in [
        "prevalence_comparison.csv",
        "model_comparison_combined.csv",
        "shap_comparison.csv",
        "wellness_hypothesis_combined.csv",
    ]:
        if not (REPORTS_DIR / req).exists():
            logger.error("Missing %s — run scripts 13-16 first", req)
            raise SystemExit(1)

    data = load_data()

    make_dashboard(data, FIGURES_DIR / "fig_phase3_dashboard.png")
    make_covid_impact_figure(data, FIGURES_DIR / "fig_covid_impact_summary.png")

    logger.info("\n══════ Phase 3 figures complete ══════")
    logger.info("Generated:")
    for f in ["fig_phase3_dashboard.png", "fig_covid_impact_summary.png"]:
        logger.info("  figures/%s", f)

    logger.info("\nAll Phase 3 scripts complete.")
    logger.info("New figures available for app: fig_prevalence_comparison, fig_riskfactor_comparison,")
    logger.info("  fig_model_comparison_forest, fig_wellness_forest_plot, fig_shap_comparison,")
    logger.info("  fig_roc_comparison, fig_phase3_dashboard, fig_covid_impact_summary")


if __name__ == "__main__":
    main()
