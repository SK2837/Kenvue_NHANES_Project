"""
Extended Hypothesis: Social and Clinical Drivers of Elevated ALT
================================================================
Goes beyond the original 9-predictor epidemiological model to test whether
social determinants and metabolic biomarkers are independent drivers of
elevated ALT after adjusting for demographics and metal exposures.

New predictors tested:
  Clinical:  Waist circumference, triglycerides, glucose, uric acid
  Behavioral: Alcohol consumption (moderate — heavy drinkers already excluded)
  Social:    Education level

Outputs:
  reports/extended_OR_table.csv           — extended model ORs
  reports/extended_model_comparison.csv   — side-by-side with original 9-predictor model
  figures/fig_extended_forest_plot.png    — forest plot (new predictors highlighted)
"""

from __future__ import annotations

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

# ALQ121 frequency codes → estimated drinking days per week
_FREQ_TO_DPW = {
    0: 0.0, 1: 7.0, 2: 6.0, 3: 3.5, 4: 2.0,
    5: 1.0, 6: 0.625, 7: 0.25, 8: 0.17, 9: 0.087, 10: 0.029,
}

# ── Predictor groups for colour-coding ───────────────────────────────────────
PREDICTOR_GROUPS = {
    "Demographics":        ["Age (years)", "Male sex", "Poverty-income ratio", "College education"],
    "Body composition":    ["Waist circumference (cm)"],
    "Metabolic (clinical)": ["Triglycerides (log)", "Glucose (log)", "Uric acid (log)"],
    "Clinical/Behavioral": ["Diabetes", "Ever smoker", "Weekly alcohol drinks"],
    "Heavy metals":        ["log(Blood Lead)", "log(Blood Cadmium)", "log(Blood Mercury)"],
}

GROUP_COLOURS = {
    "Demographics":         "#4C72B0",
    "Body composition":     "#55A868",
    "Metabolic (clinical)": "#E67E22",
    "Clinical/Behavioral":  "#C44E52",
    "Heavy metals":         "#8172B2",
}

NEW_PREDICTOR_LABELS = {
    "log_triglycerides", "log_glucose", "log_uric_acid",
    "weekly_drinks", "college_edu",
}


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ── Original features (keep for comparison) ───────────────────────────────
    out["log_lead"]    = np.log(out["LBXBPB"].clip(lower=0.001))
    out["log_cadmium"] = np.log(out["LBXBCD"].clip(lower=0.001))
    out["log_mercury"] = np.log(out["LBXTHG"].clip(lower=0.001))
    out["sex_male"]    = (out["RIAGENDR"] == 1).astype(float)
    out["diabetes"]    = (out["DIQ010"] == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan
    out["ever_smoker"] = (out["SMQ020"] == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan

    # ── New clinical features ─────────────────────────────────────────────────

    # 1. Waist circumference (replacing BMI — more specific to visceral fat)
    #    BMXWAIST already numeric in cm — use directly

    # 2. Triglycerides (LBXSTR, mg/dL) — log-transform (right-skewed)
    out["log_triglycerides"] = np.log(out["LBXSTR"].clip(lower=1))

    # 3. Glucose (LBXSGL, mg/dL) — log-transform; captures pre-diabetic spectrum
    out["log_glucose"] = np.log(out["LBXSGL"].clip(lower=1))

    # 4. Uric acid (LBXSUA, mg/dL) — metabolic syndrome / gout / NAFLD marker
    out["log_uric_acid"] = np.log(out["LBXSUA"].clip(lower=0.1))

    # ── New behavioral feature ────────────────────────────────────────────────

    # 5. Weekly alcohol drinks (moderate — heavy drinkers were excluded in 02_build_dataset.py)
    #    ALQ121 (frequency) × ALQ130 (drinks per sitting)
    freq_dpw   = out["ALQ121"].map(_FREQ_TO_DPW)
    drinks_day = out["ALQ130"].clip(lower=0, upper=15)
    out["weekly_drinks"] = (freq_dpw * drinks_day).fillna(0)
    # Non-drinkers (ALQ111==2 → never had a drink): set to 0
    never = out["ALQ111"] == 2
    out.loc[never, "weekly_drinks"] = 0.0
    # Log-plus-one to handle zero inflated distribution
    out["log_weekly_drinks"] = np.log1p(out["weekly_drinks"])

    # ── New social feature ────────────────────────────────────────────────────

    # 6. Education: college or above = 1 (DMDEDUC2: 4=some college, 5=college grad)
    #    Refused (7) / Don't know (9) → NaN
    out["college_edu"] = np.where(
        out["DMDEDUC2"].isin([7, 9]) | out["DMDEDUC2"].isna(), np.nan,
        (out["DMDEDUC2"] >= 4).astype(float)
    )

    return out


# Column labels for the extended predictor set
EXTENDED_PREDICTORS = [
    "RIDAGEYR", "sex_male", "INDFMPIR",
    "BMXWAIST",           # replaces BMI — more specific to visceral fat
    "diabetes",           # kept as clinical diagnosis (not replaced by glucose to avoid collinearity)
    "ever_smoker",
    "log_weekly_drinks",  # NEW — alcohol (moderate; heavy already excluded)
    "log_triglycerides",  # NEW — metabolic syndrome marker
    "college_edu",        # NEW — social determinant
    "log_lead", "log_cadmium", "log_mercury",
]
# NOTE: glucose and uric acid dropped — they are collinear with diabetes + triglycerides
# and caused extreme CI widening (VIF >> 10) in preliminary runs.

EXTENDED_COL_LABELS = {
    "RIDAGEYR":           "Age (years)",
    "sex_male":           "Male sex",
    "INDFMPIR":           "Poverty-income ratio",
    "BMXWAIST":           "Waist circumference (cm)",
    "diabetes":           "Diabetes",
    "ever_smoker":        "Ever smoker",
    "log_weekly_drinks":  "Weekly alcohol drinks (log)",
    "log_triglycerides":  "Triglycerides (log)",
    "college_edu":        "College education",
    "log_lead":           "log(Blood Lead)",
    "log_cadmium":        "log(Blood Cadmium)",
    "log_mercury":        "log(Blood Mercury)",
}

# Original 9-predictor labels for comparison column
ORIGINAL_PREDICTORS = [
    "RIDAGEYR", "sex_male", "INDFMPIR", "BMXBMI",
    "diabetes", "ever_smoker",
    "log_lead", "log_cadmium", "log_mercury",
]
ORIGINAL_COL_LABELS = {
    "RIDAGEYR":   "Age (years)",
    "sex_male":   "Male sex",
    "INDFMPIR":   "Poverty-income ratio",
    "BMXBMI":     "BMI (kg/m²)",
    "diabetes":   "Diabetes",
    "ever_smoker":"Ever smoker",
    "log_lead":   "log(Blood Lead)",
    "log_cadmium":"log(Blood Cadmium)",
    "log_mercury":"log(Blood Mercury)",
}


# ── Survey-weighted logistic regression ───────────────────────────────────────

def run_svy_model(
    df: pd.DataFrame,
    predictors: list[str],
    outcome: str = "ALT_elevated_40",
    col_labels: dict | None = None,
) -> pd.DataFrame:
    model_cols = [outcome, "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] + predictors
    model_df   = df[model_cols].dropna()
    n          = len(model_df)
    n_events   = int(model_df[outcome].sum())
    epv        = n_events / len(predictors)
    logger.info("Sample: %d rows | Events: %d | EPV: %.1f", n, n_events, epv)
    if epv < 10:
        logger.warning("EPV=%.1f < 10: model may be unstable. Interpret cautiously.", epv)

    pl_df  = pl.from_pandas(model_df)
    design = svy.Design(stratum="SDMVSTRA", psu="SDMVPSU", wgt="WTMEC2YR")
    sample = svy.Sample(data=pl_df, design=design)
    fit    = sample.glm.fit(y=outcome, x=predictors, family="binomial", link="logit")

    rows = []
    for coef in fit.coefs:
        if coef.term == "_intercept_":
            continue
        beta = float(coef.est)
        lo   = float(coef.lci)
        hi   = float(coef.uci)
        pval = float(coef.wald.p_value) if coef.wald is not None else np.nan
        label = (col_labels or {}).get(coef.term, coef.term)
        rows.append({
            "variable": coef.term,
            "label":    label,
            "beta":     round(beta, 4),
            "OR":       round(np.exp(beta), 3),
            "CI_low":   round(np.exp(lo), 3),
            "CI_high":  round(np.exp(hi), 3),
            "p_value":  round(pval, 4),
        })

    result = pd.DataFrame(rows)
    result["significant"] = result["p_value"] < 0.05
    return result


# ── Forest plot ───────────────────────────────────────────────────────────────

def make_extended_forest_plot(
    or_df: pd.DataFrame,
    new_predictor_labels: set[str],
    title: str,
    path: Path,
) -> None:
    label_to_group = {
        lbl: grp
        for grp, lbls in PREDICTOR_GROUPS.items()
        for lbl in lbls
    }
    or_df = or_df.copy()
    or_df["group"]  = or_df["label"].map(label_to_group).fillna("Other")
    or_df["colour"] = or_df["group"].map(GROUP_COLOURS).fillna("#888888")
    or_df["is_new"] = or_df["variable"].isin(new_predictor_labels)
    or_df = or_df.dropna(subset=["CI_low", "CI_high"])

    group_order = list(PREDICTOR_GROUPS.keys())
    or_df["group_rank"] = or_df["group"].apply(
        lambda g: group_order.index(g) if g in group_order else len(group_order)
    )
    or_df = or_df.sort_values("group_rank", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(7, len(or_df) * 0.5)))

    for i, row in or_df.iterrows():
        marker    = "D" if row["is_new"] else "o"
        markersize = 8 if row["is_new"] else 6
        lw        = 2.0 if row["is_new"] else 1.5
        alpha     = 1.0 if row["is_new"] else 0.8

        ax.errorbar(
            x=row["OR"], y=i,
            xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
            fmt=marker, color=row["colour"],
            markersize=markersize, capsize=4,
            linewidth=lw, elinewidth=lw, alpha=alpha,
        )

        sig_marker = " *" if row["p_value"] < 0.05 else ""
        ax.text(
            max(or_df["CI_high"]) * 1.02, i,
            f"{row['OR']:.2f} ({row['CI_low']:.2f}–{row['CI_high']:.2f}){sig_marker}",
            va="center", fontsize=7.5, color="#333333",
            fontweight="bold" if row["is_new"] else "normal",
        )

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(or_df)))
    ax.set_yticklabels(
        [f"★ {r['label']}" if r["is_new"] else r["label"] for _, r in or_df.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Odds Ratio (95% CI) — survey-weighted, Taylor-series CIs", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)

    legend_patches = [
        mpatches.Patch(color=col, label=grp)
        for grp, col in GROUP_COLOURS.items()
    ]
    legend_patches += [
        mpatches.Patch(facecolor="white", edgecolor="black", label="○ Original predictor"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="★ New predictor  (* p<0.05)", linewidth=1.5),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7.5, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Comparison table ──────────────────────────────────────────────────────────

def make_comparison_table(
    orig_df: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> pd.DataFrame:
    orig_sub = orig_df[["label", "OR", "CI_low", "CI_high", "p_value"]].copy()
    orig_sub.columns = ["label", "OR_orig", "CI_low_orig", "CI_high_orig", "p_orig"]

    ext_sub  = ext_df[["label", "OR", "CI_low", "CI_high", "p_value", "variable"]].copy()
    ext_sub.columns  = ["label", "OR_ext", "CI_low_ext", "CI_high_ext", "p_ext", "variable"]

    comp = ext_sub.merge(orig_sub, on="label", how="left")
    comp["is_new"] = comp["variable"].isin(NEW_PREDICTOR_LABELS | {"BMXWAIST", "log_weekly_drinks", "log_triglycerides", "log_glucose", "log_uric_acid", "college_edu"})
    comp["sig_ext"]  = comp["p_ext"].apply(lambda p: "✓" if p < 0.05 else "")
    comp["sig_orig"] = comp["p_orig"].apply(lambda p: "✓" if p < 0.05 else "—" if pd.isna(p) else "")

    cols = ["label", "is_new",
            "OR_orig", "CI_low_orig", "CI_high_orig", "p_orig", "sig_orig",
            "OR_ext",  "CI_low_ext",  "CI_high_ext",  "p_ext",  "sig_ext"]
    return comp[[c for c in cols if c in comp.columns]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        logger.error("Analytic table not found — run 02_build_dataset.py first")
        raise SystemExit(1)

    df = pd.read_parquet(parquet)
    logger.info("Loaded: %d rows × %d cols", len(df), df.shape[1])

    df = prepare_extended_features(df)

    # ── Model A: Original 9 predictors (reproduction) ─────────────────────────
    logger.info("\n─── Model A: Original 9-predictor model ───")
    orig_or = run_svy_model(df, ORIGINAL_PREDICTORS, col_labels=ORIGINAL_COL_LABELS)
    logger.info("\n%s", orig_or[["label", "OR", "CI_low", "CI_high", "p_value"]].to_string(index=False))

    # ── Model B: Extended 14-predictor model ──────────────────────────────────
    logger.info("\n─── Model B: Extended model (social + clinical additions) ───")
    ext_or = run_svy_model(df, EXTENDED_PREDICTORS, col_labels=EXTENDED_COL_LABELS)
    logger.info("\n%s", ext_or[["label", "OR", "CI_low", "CI_high", "p_value"]].to_string(index=False))

    # ── Save outputs ───────────────────────────────────────────────────────────
    ext_or.to_csv(REPORTS_DIR / "extended_OR_table.csv", index=False)
    logger.info("Saved reports/extended_OR_table.csv")

    comp = make_comparison_table(orig_or, ext_or)
    comp.to_csv(REPORTS_DIR / "extended_model_comparison.csv", index=False)
    logger.info("Saved reports/extended_model_comparison.csv")

    make_extended_forest_plot(
        ext_or,
        new_predictor_labels={
            "log_weekly_drinks", "log_triglycerides", "college_edu", "BMXWAIST",
        },
        title=(
            "Extended Model: Social + Clinical Drivers of Elevated ALT — NHANES 2017–2018\n"
            "(★ = new predictors beyond original model  |  * p<0.05  |  N restricted to complete cases)"
        ),
        path=FIGURES_DIR / "fig_extended_forest_plot.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n══════ HYPOTHESIS TEST SUMMARY ══════")
    new_vars = ext_or[ext_or["variable"].isin({
        "log_weekly_drinks", "log_triglycerides", "college_edu", "BMXWAIST",
    })]
    sig_new  = new_vars[new_vars["p_value"] < 0.05]
    ns_new   = new_vars[new_vars["p_value"] >= 0.05]

    logger.info("\nSIGNIFICANT new predictors (p < 0.05):")
    for _, r in sig_new.iterrows():
        logger.info("  ✓ %-35s OR=%.3f  p=%.4f", r["label"], r["OR"], r["p_value"])

    logger.info("\nNon-significant new predictors:")
    for _, r in ns_new.iterrows():
        logger.info("  ✗ %-35s OR=%.3f  p=%.4f", r["label"], r["OR"], r["p_value"])

    logger.info(
        "\nConclusion: %d of %d new predictors reach p<0.05 after adjustment.",
        len(sig_new), len(new_vars)
    )
    logger.info("Extended analysis complete.")


if __name__ == "__main__":
    main()
