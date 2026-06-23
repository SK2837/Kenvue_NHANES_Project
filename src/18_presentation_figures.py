"""
Assemble presentation-ready figures into figures/presentation_ready/.

Creates two new figures:
  fig_model_auc_comparison.png  — clean (0.77) vs leaky benchmark (0.97) side-by-side
  fig_agent_architecture.png    — agentic AI system diagram

Copies all other relevant figures from phase1_eda, phase2_analysis, phase3_covid.
"""

import json
import logging
import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
FIGURES     = ROOT / "figures"
REPORTS     = ROOT / "reports" / "phase2_analysis"
OUT         = FIGURES / "presentation_ready"
OUT.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def copy_figure(src: Path, dst_name: str) -> None:
    if src.exists():
        shutil.copy2(src, OUT / dst_name)
        logger.info("Copied  %s", dst_name)
    else:
        logger.warning("MISSING %s — skipped", src)


# ── Figure 1: Model AUC comparison bar ───────────────────────────────────────

def make_auc_comparison() -> None:
    metrics_clean = json.loads((REPORTS / "ml_metrics.json").read_text())
    metrics_bench = json.loads((REPORTS / "ml_metrics_benchmark.json").read_text())

    auc_clean = metrics_clean["roc_auc"]
    auc_bench = metrics_bench["roc_auc"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        ["Clean risk-factor model\n(demographics + metabolic\n+ heavy metals)",
         "Full biochemistry benchmark*\n(includes liver co-travelers:\nAST, GGT, ALP…)"],
        [auc_clean, auc_bench],
        color=["#4C72B0", "#C44E52"],
        width=0.45,
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, val in zip(bars, [auc_clean, auc_bench]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"AUC = {val:.3f}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUC-ROC (held-out test set, N=709)", fontsize=11)
    ax.set_title(
        "XGBoost Model Performance: Honest vs Leaky",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="Random classifier")
    ax.legend(fontsize=9, loc="lower right")

    ax.text(
        0.5, -0.22,
        "* Benchmark AUC is inflated — AST alone accounts for >50% of SHAP importance.\n"
        "  The clean model (AUC 0.77) uses only actionable risk factors and avoids target leakage.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#555",
    )

    plt.tight_layout()
    p = OUT / "fig_model_auc_comparison.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Created %s", p.name)


# ── Figure 2: Agent architecture diagram ─────────────────────────────────────

def _box(ax, x, y, w, h, text, facecolor, edgecolor="#333", fontsize=9,
         text_color="white", bold=False):
    fancy = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4, zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold" if bold else "normal",
            zorder=4, wrap=True, multialignment="center")


def _arrow(ax, x1, y1, x2, y2, label="", color="#555"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.015, my, label, fontsize=7.5, color=color, va="center")


def make_agent_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    ax.set_title(
        "Agentic AI System Architecture — NHANES Liver Risk Assistant",
        fontsize=13, fontweight="bold", pad=14, color="#1a1a2e",
    )

    # ── User / UI layer ───────────────────────────────────────────────────────
    _box(ax, 0.5, 0.88, 0.38, 0.09,
         "User Interface\nStreamlit Web App  |  CLI (--cli flag)",
         facecolor="#1a1a2e", fontsize=9, bold=True)

    # ── Orchestrator ──────────────────────────────────────────────────────────
    _box(ax, 0.5, 0.68, 0.34, 0.09,
         "Orchestrator  (Claude Sonnet)\nIntent routing: qa | analysis | direct",
         facecolor="#2c6fad", fontsize=9, bold=True)
    _arrow(ax, 0.5, 0.835, 0.5, 0.725, color="#2c6fad")

    # ── Q&A Agent ─────────────────────────────────────────────────────────────
    _box(ax, 0.22, 0.47, 0.30, 0.09,
         "Q&A Agent  (Claude Haiku)\nPersonalised risk · Stats lookup\nFast, low-latency",
         facecolor="#2e8b57", fontsize=8.5)
    _arrow(ax, 0.36, 0.68, 0.27, 0.515, label="qa intent", color="#2e8b57")

    # ── Analysis Agent ────────────────────────────────────────────────────────
    _box(ax, 0.78, 0.47, 0.30, 0.09,
         "Analysis Agent  (Claude Sonnet)\nSubgroup analysis · Explanations\nDeep reasoning",
         facecolor="#8b4513", fontsize=8.5)
    _arrow(ax, 0.64, 0.68, 0.73, 0.515, label="analysis intent", color="#8b4513")

    # ── Tool boxes (Q&A side) ─────────────────────────────────────────────────
    qa_tools = [
        (0.08, 0.26, "compute_individual_risk"),
        (0.22, 0.26, "lookup_or_table"),
        (0.36, 0.26, "prevalence_by_group"),
    ]
    for tx, ty, tlabel in qa_tools:
        _box(ax, tx, ty, 0.135, 0.065, tlabel,
             facecolor="#d4edda", edgecolor="#2e8b57",
             text_color="#155724", fontsize=7.5)
        _arrow(ax, 0.22, 0.425, tx, ty + 0.033, color="#2e8b57")

    # ── Tool boxes (Analysis side) ────────────────────────────────────────────
    an_tools = [
        (0.635, 0.26, "run_subgroup_analysis"),
        (0.78,  0.26, "query_analytic_table"),
        (0.925, 0.26, "explain_risk_factors"),
    ]
    for tx, ty, tlabel in an_tools:
        _box(ax, tx, ty, 0.135, 0.065, tlabel,
             facecolor="#fde8d8", edgecolor="#8b4513",
             text_color="#7b2d00", fontsize=7.5)
        _arrow(ax, 0.78, 0.425, tx, ty + 0.033, color="#8b4513")

    # ── Shared tools (bottom centre) ──────────────────────────────────────────
    shared = [
        (0.28, 0.08, "read_shap_rankings"),
        (0.42, 0.08, "read_ml_metrics"),
        (0.56, 0.08, "read_methods_report"),
        (0.70, 0.08, "wellness_hypothesis_results"),
    ]
    for tx, ty, tlabel in shared:
        _box(ax, tx, ty, 0.135, 0.065, tlabel,
             facecolor="#e8eaf6", edgecolor="#3949ab",
             text_color="#1a237e", fontsize=7.5)

    # label shared row
    ax.text(0.11, 0.08, "Shared\ntools:", fontsize=8, color="#3949ab",
            va="center", ha="center", fontweight="bold")

    # arrows from both agents to shared tools
    for tx, ty, _ in shared:
        _arrow(ax, 0.22, 0.425, tx, ty + 0.033, color="#aaa")
        _arrow(ax, 0.78, 0.425, tx, ty + 0.033, color="#aaa")

    # ── Data layer label ──────────────────────────────────────────────────────
    _box(ax, 0.5, 0.08, 0.90, 0.005,
         "", facecolor="#cfd8dc", edgecolor="#90a4ae")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (mpatches.Patch(facecolor="#2e8b57", label="Q&A Agent tools")),
        (mpatches.Patch(facecolor="#8b4513", label="Analysis Agent tools")),
        (mpatches.Patch(facecolor="#3949ab", label="Shared / read-only tools")),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              bbox_to_anchor=(0.0, 0.0), framealpha=0.9)

    # tool count note
    ax.text(0.98, 0.01, "11 tool functions total", ha="right", va="bottom",
            fontsize=8, color="#555", style="italic", transform=ax.transAxes)

    plt.tight_layout()
    p = OUT / "fig_agent_architecture.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Created %s", p.name)


# ── Copy all relevant existing figures ───────────────────────────────────────

COPY_MAP = {
    # Phase 1 EDA
    FIGURES / "phase1_eda" / "fig_alt_distribution.png":     "01_fig_alt_distribution.png",
    FIGURES / "phase1_eda" / "fig_bmi_by_alt.png":           "02_fig_bmi_by_alt.png",
    FIGURES / "phase1_eda" / "fig_correlation_heatmap.png":  "03_fig_correlation_heatmap.png",
    FIGURES / "phase1_eda" / "fig_metals_distribution.png":  "04_fig_metals_distribution.png",
    FIGURES / "phase1_eda" / "fig_prevalence_by_race.png":   "05_fig_prevalence_by_race.png",
    # Phase 2 inference
    FIGURES / "phase2_analysis" / "fig_forest_plot_OR.png":       "06_fig_forest_plot_OR.png",
    FIGURES / "phase2_analysis" / "fig_roc_calibration.png":      "07_fig_roc_calibration.png",
    FIGURES / "phase2_analysis" / "fig_shap_importance.png":      "08_fig_shap_importance_clean.png",
    FIGURES / "phase2_analysis" / "fig_shap_summary.png":         "09_fig_shap_summary_clean.png",
    FIGURES / "phase2_analysis" / "fig_shap_waterfall_high.png":  "10_fig_shap_waterfall_high.png",
    FIGURES / "phase2_analysis" / "fig_shap_waterfall_low.png":   "11_fig_shap_waterfall_low.png",
    FIGURES / "phase2_analysis" / "fig_triglycerides_by_alt.png": "12_fig_triglycerides_by_alt.png",
    FIGURES / "phase2_analysis" / "fig_waist_by_alt.png":         "13_fig_waist_by_alt.png",
    # Phase 3 cohort comparison
    FIGURES / "phase3_covid" / "fig_prevalence_comparison.png":   "14_fig_prevalence_comparison.png",
    FIGURES / "phase3_covid" / "fig_covid_impact_summary.png":    "15_fig_cohort_impact_summary.png",
    FIGURES / "phase3_covid" / "fig_phase3_dashboard.png":        "16_fig_cohort_dashboard.png",
    FIGURES / "phase3_covid" / "fig_riskfactor_comparison.png":   "17_fig_riskfactor_comparison.png",
    FIGURES / "phase3_covid" / "fig_roc_comparison.png":          "18_fig_roc_comparison.png",
    FIGURES / "phase3_covid" / "fig_shap_comparison.png":         "19_fig_shap_comparison.png",
    FIGURES / "phase3_covid" / "fig_wellness_forest_plot.png":    "20_fig_wellness_forest_plot.png",
}


def main() -> None:
    logger.info("Creating presentation_ready figures → %s", OUT)

    make_auc_comparison()
    make_agent_architecture()

    for src, dst in COPY_MAP.items():
        copy_figure(src, dst)

    # Write an index
    index_lines = ["# Presentation-Ready Figures\n",
                   "Numbered in suggested slide order.\n\n",
                   "| File | Description |\n",
                   "|------|-------------|\n"]
    descriptions = {
        "01": "ALT distribution (population histogram)",
        "02": "BMI by ALT status",
        "03": "Correlation heatmap (full biochemistry panel)",
        "04": "Heavy metals distribution",
        "05": "Weighted ALT prevalence by race/ethnicity",
        "06": "Forest plot — adjusted ORs, 10 predictors (primary inference result)",
        "07": "ROC + calibration — clean model (0.77) vs leaky benchmark (0.97)",
        "08": "SHAP importance — clean model (top drivers: age, triglycerides, iron, glucose, BMI)",
        "09": "SHAP beeswarm — clean model",
        "10": "SHAP waterfall — highest-risk individual",
        "11": "SHAP waterfall — lowest-risk individual",
        "12": "Triglycerides by ALT status",
        "13": "Waist circumference by ALT status",
        "14": "Prevalence comparison 2017-2018 vs 2021-2023",
        "15": "Cohort comparison narrative (corrected — no causal COVID language)",
        "16": "Full cohort comparison dashboard",
        "17": "Risk factor OR comparison across cohorts",
        "18": "ROC comparison across cohorts",
        "19": "SHAP comparison across cohorts",
        "20": "Wellness variable forest plot",
        "NEW_auc": "Model AUC comparison: clean 0.77 vs benchmark 0.97 (new)",
        "NEW_arch": "Agent architecture diagram (new)",
    }
    for src, dst in COPY_MAP.items():
        prefix = dst[:2]
        desc = descriptions.get(prefix, "")
        index_lines.append(f"| {dst} | {desc} |\n")
    index_lines.append(f"| fig_model_auc_comparison.png | {descriptions['NEW_auc']} |\n")
    index_lines.append(f"| fig_agent_architecture.png | {descriptions['NEW_arch']} |\n")

    (OUT / "INDEX.md").write_text("".join(index_lines))
    logger.info("Written INDEX.md")

    total = len(list(OUT.glob("*.png")))
    logger.info("Done — %d figures in %s", total, OUT)


if __name__ == "__main__":
    main()
