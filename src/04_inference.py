"""
Survey-weighted logistic regression for the NHANES 2017-2018 liver-injury screen.

Uses Taylor-series linearization via the `svy` package to produce design-correct
odds ratios and 95% CIs that represent the U.S. adult population — not just the
3,543-person sample.

Outputs:
  reports/inference_OR_table.csv              (primary: sex-specific AASLD threshold)
  reports/inference_OR_table_sensitivity.csv  (sensitivity: unisex 40 U/L threshold)
  figures/fig_forest_plot_OR.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polars as pl
import svy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase2_analysis"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase2_analysis"

# ── Predictor groups (used to colour forest plot) ─────────────────────────────
PREDICTOR_GROUPS = {
    "Demographics":    ["Age (years)", "Male sex", "Poverty-income ratio"],
    "Anthropometrics": ["Waist circumference (cm)"],
    "Clinical":        ["Diabetes", "Ever smoker", "Triglycerides (log)"],
    "Heavy Metals":    ["log(Blood Lead)", "log(Blood Cadmium)", "log(Blood Mercury)"],
}

GROUP_COLOURS = {
    "Demographics":    "#4C72B0",
    "Anthropometrics": "#55A868",
    "Clinical":        "#C44E52",
    "Heavy Metals":    "#8172B2",
}


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Log-transform right-skewed metals (all positive, no zeros)
    out["log_lead"]    = np.log(out["LBXBPB"])
    out["log_cadmium"] = np.log(out["LBXBCD"])
    out["log_mercury"] = np.log(out["LBXTHG"])

    # Binary sex: 1=male, 0=female
    out["sex_male"] = (out["RIAGENDR"] == 1).astype(float)

    # Race/ethnicity dummies (reference: Non-Hispanic White = RIDRETH3 == 3)
    out["race_mex_am"]    = (out["RIDRETH3"] == 1).astype(float)
    out["race_oth_hisp"]  = (out["RIDRETH3"] == 2).astype(float)
    out["race_nh_black"]  = (out["RIDRETH3"] == 4).astype(float)
    out["race_nh_asian"]  = (out["RIDRETH3"] == 6).astype(float)
    out["race_oth_multi"] = (out["RIDRETH3"] == 7).astype(float)

    # Diabetes binary (1=diagnosed, 0=no/borderline)
    out["diabetes"] = (out["DIQ010"] == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan

    # Smoking binary (1=ever smoked 100+ cigs, 0=never)
    out["ever_smoker"] = (out["SMQ020"] == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan

    # Triglycerides: log-transform to reduce right skew (LBXSTR in mg/dL)
    out["log_triglycerides"] = np.log(out["LBXSTR"].clip(lower=1))

    return out


# Column names used in the svy model → human labels for the OR table
# Updated model (10 predictors): BMXWAIST replaces BMXBMI; log_triglycerides added
COL_LABELS = {
    "RIDAGEYR":          "Age (years)",
    "sex_male":          "Male sex",
    "INDFMPIR":          "Poverty-income ratio",
    "BMXWAIST":          "Waist circumference (cm)",
    "diabetes":          "Diabetes",
    "ever_smoker":       "Ever smoker",
    "log_triglycerides": "Triglycerides (log)",
    "log_lead":          "log(Blood Lead)",
    "log_cadmium":       "log(Blood Cadmium)",
    "log_mercury":       "log(Blood Mercury)",
}

PREDICTORS = list(COL_LABELS.keys())


# ── Survey-weighted logistic regression ───────────────────────────────────────

def run_model(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Fit survey-weighted logistic regression; return OR table."""
    model_cols = [outcome, "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] + PREDICTORS
    model_df   = df[model_cols].dropna()
    logger.info("Model sample (complete cases): %s rows", f"{len(model_df):,}")

    pl_df  = pl.from_pandas(model_df)
    design = svy.Design(stratum="SDMVSTRA", psu="SDMVPSU", wgt="WTMEC2YR")
    sample = svy.Sample(data=pl_df, design=design)

    fit = sample.glm.fit(
        y=outcome,
        x=PREDICTORS,
        family="binomial",
        link="logit",
    )

    # fit.coefs is a list of GLMCoef(term, est, se, lci, uci, wald)
    rows = []
    for coef in fit.coefs:
        if coef.term == "_intercept_":
            continue
        beta = float(coef.est)
        lo   = float(coef.lci)
        hi   = float(coef.uci)
        pval = float(coef.wald.p_value) if coef.wald is not None else np.nan

        rows.append({
            "variable": coef.term,
            "label":    COL_LABELS.get(coef.term, coef.term),
            "beta":     round(beta, 4),
            "OR":       round(np.exp(beta), 3),
            "CI_low":   round(np.exp(lo), 3),
            "CI_high":  round(np.exp(hi), 3),
            "p_value":  round(pval, 4) if not np.isnan(pval) else np.nan,
        })

    return pd.DataFrame(rows)


# ── Forest plot ───────────────────────────────────────────────────────────────

def make_forest_plot(or_df: pd.DataFrame, title: str, path: Path) -> None:
    # Assign colour by group
    label_to_group = {lbl: grp for grp, lbls in PREDICTOR_GROUPS.items() for lbl in lbls}
    or_df = or_df.copy()
    or_df["group"]  = or_df["label"].map(label_to_group).fillna("Other")
    or_df["colour"] = or_df["group"].map(GROUP_COLOURS).fillna("#888888")

    # Drop rows with missing CIs (shouldn't happen but guard)
    or_df = or_df.dropna(subset=["CI_low", "CI_high"])

    # Sort: demographics first, metals highlighted near bottom
    group_order = list(PREDICTOR_GROUPS.keys())
    or_df["group_rank"] = or_df["group"].apply(
        lambda g: group_order.index(g) if g in group_order else len(group_order)
    )
    or_df = or_df.sort_values("group_rank", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(6, len(or_df) * 0.45)))

    for i, row in or_df.iterrows():
        ax.errorbar(
            x=row["OR"],
            y=i,
            xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
            fmt="o",
            color=row["colour"],
            markersize=6,
            capsize=4,
            linewidth=1.5,
            elinewidth=1.5,
        )
        # Annotate OR value
        ax.text(
            max(or_df["CI_high"]) * 1.02, i,
            f"{row['OR']:.2f} ({row['CI_low']:.2f}–{row['CI_high']:.2f})",
            va="center", fontsize=7.5, color="#333333",
        )

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(or_df)))
    ax.set_yticklabels(or_df["label"], fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI) — design-correct Taylor-series linearization", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)

    # Legend
    legend_patches = [
        mpatches.Patch(color=col, label=grp)
        for grp, col in GROUP_COLOURS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        logger.error("Analytic table not found — run 02_build_dataset.py first")
        raise SystemExit(1)

    df = pd.read_parquet(parquet)
    logger.info("Loaded analytic table: %s rows × %s cols", f"{len(df):,}", df.shape[1])

    df = prepare_features(df)

    # ── Primary model: unisex 40 U/L threshold ───────────────────────────────
    # Uses ALT_elevated_40 so sex is a clean covariate (no threshold interaction)
    # Updated 10-predictor spec: waist replaces BMI; log_triglycerides added
    logger.info("Fitting primary model (ALT > 40 U/L unisex, 10 predictors) ...")
    or_primary = run_model(df, "ALT_elevated_40")

    out_primary = REPORTS_DIR / "inference_OR_table.csv"
    or_primary.to_csv(out_primary, index=False)
    logger.info("Saved %s", out_primary)

    logger.info("\nPrimary model odds ratios:\n%s",
                or_primary[["label", "OR", "CI_low", "CI_high", "p_value"]].to_string(index=False))

    make_forest_plot(
        or_primary,
        title="Adjusted ORs for Elevated ALT (>40 U/L) — NHANES 2017–2018\n(Survey-weighted logistic regression; Taylor-series CIs; 10 predictors; N=3,543)",
        path=FIGURES_DIR / "fig_forest_plot_OR.png",
    )

    # ── Sensitivity model: sex-specific AASLD thresholds (no sex predictor) ─
    logger.info("Fitting sensitivity model (sex-specific AASLD thresholds, no sex predictor) ...")
    # Drop sex_male from predictors for this model since outcome is sex-defined
    sens_labels = {k: v for k, v in COL_LABELS.items() if k != "sex_male"}
    sens_preds  = list(sens_labels.keys())

    model_cols  = ["ALT_elevated", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] + sens_preds
    sens_df     = df[model_cols].dropna()
    pl_sens     = pl.from_pandas(sens_df)
    design_s    = svy.Design(stratum="SDMVSTRA", psu="SDMVPSU", wgt="WTMEC2YR")
    sample_s    = svy.Sample(data=pl_sens, design=design_s)
    fit_s       = sample_s.glm.fit(y="ALT_elevated", x=sens_preds, family="binomial", link="logit")

    rows_s = []
    for coef in fit_s.coefs:
        if coef.term == "_intercept_":
            continue
        beta = float(coef.est)
        rows_s.append({
            "variable": coef.term,
            "label":    sens_labels.get(coef.term, coef.term),
            "OR":       round(np.exp(beta), 3),
            "CI_low":   round(np.exp(float(coef.lci)), 3),
            "CI_high":  round(np.exp(float(coef.uci)), 3),
            "p_value":  round(float(coef.wald.p_value), 4) if coef.wald else np.nan,
        })
    or_sensitivity = pd.DataFrame(rows_s)

    out_sens = REPORTS_DIR / "inference_OR_table_sensitivity.csv"
    or_sensitivity.to_csv(out_sens, index=False)
    logger.info("Saved %s", out_sens)

    # ── Directional sanity check ──────────────────────────────────────────────
    logger.info("\n── Sanity checks ──")
    waist_or = or_primary.loc[or_primary["variable"] == "BMXWAIST", "OR"].values
    if len(waist_or):
        direction = "✓ correct (↑waist → ↑ALT risk)" if waist_or[0] > 1 else "✗ unexpected direction"
        logger.info("Waist OR: %.3f  %s", waist_or[0], direction)

    trig_or = or_primary.loc[or_primary["variable"] == "log_triglycerides", "OR"].values
    if len(trig_or):
        direction = "✓ correct (↑triglycerides → ↑ALT risk)" if trig_or[0] > 1 else "✗ unexpected direction"
        logger.info("Triglycerides OR: %.3f  %s", trig_or[0], direction)

    lead_or = or_primary.loc[or_primary["variable"] == "log_lead", "OR"].values
    if len(lead_or):
        direction = "✓ expected direction" if lead_or[0] > 1 else "↓ inverse (worth investigating)"
        logger.info("log(Lead) OR: %.3f  %s", lead_or[0], direction)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
