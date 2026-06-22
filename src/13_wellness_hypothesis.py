"""
Wellness Variable Hypothesis Test — Pre/Post COVID Comparison
=============================================================
Tests whether PHQ-9 depression, short sleep (<7h), and high sedentary
activity (≥8h/day) are independent predictors of elevated ALT, in both
the 2017-2018 (pre-COVID) and 2021-2023 (post-COVID) NHANES cohorts.

Analysis structure:
  1. Unadjusted (univariate): each wellness variable alone
  2. Adjusted: added to the 10-predictor epidemiologic model
  3. Side-by-side comparison across both cohorts

Outputs:
  reports/wellness_hypothesis_2017.csv   — 2017-2018 results
  reports/wellness_hypothesis_2021.csv   — 2021-2023 results
  reports/wellness_hypothesis_combined.csv — both cohorts, wide format
  figures/fig_wellness_forest_plot.png   — side-by-side forest plot
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
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase3_covid"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase3_covid"

# ── 10-predictor base model (same as 04_inference.py) ────────────────────────
BASE_PREDICTORS = [
    "RIDAGEYR", "sex_male", "INDFMPIR", "BMXWAIST",
    "diabetes", "ever_smoker", "log_triglycerides",
    "log_lead", "log_cadmium", "log_mercury",
]

WELLNESS_VARS = {
    "phq9_dep":      "Depression (PHQ-9 ≥10)",
    "short_sleep":   "Short sleep (<7 h/night)",
    "high_sedentary":"High sedentary (≥8 h/day)",
}

OUTCOME = "ALT_elevated_40"


# ── Feature engineering (same as 04_inference.py) ────────────────────────────

def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_lead"]         = np.log(out["LBXBPB"].clip(lower=0.001))
    out["log_cadmium"]      = np.log(out["LBXBCD"].clip(lower=0.001))
    out["log_mercury"]      = np.log(out["LBXTHG"].clip(lower=0.001))
    out["log_triglycerides"]= np.log(out["LBXSTR"].clip(lower=1))
    out["sex_male"]   = (out["RIAGENDR"] == 1).astype(float)
    out["diabetes"]   = (out["DIQ010"] == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan
    out["ever_smoker"]= (out["SMQ020"] == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan
    return out


# ── Survey-weighted logistic regression helper ────────────────────────────────

def run_svy_logistic(
    df: pd.DataFrame,
    predictors: list[str],
    outcome: str = OUTCOME,
    label: str = "",
) -> pd.DataFrame:
    cols = [outcome, "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] + predictors
    model_df = df[cols].dropna()
    n        = len(model_df)
    n_events = int(model_df[outcome].sum())
    epv      = n_events / len(predictors)
    logger.info(
        "[%s] N=%d  events=%d  EPV=%.1f  predictors=%d",
        label, n, n_events, epv, len(predictors),
    )
    if epv < 10:
        logger.warning("EPV=%.1f < 10 — interpret cautiously", epv)

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
        rows.append({
            "variable": coef.term,
            "beta":     round(beta, 4),
            "OR":       round(np.exp(beta), 3),
            "CI_low":   round(np.exp(lo), 3),
            "CI_high":  round(np.exp(hi), 3),
            "p_value":  round(pval, 4),
            "n":        n,
            "n_events": n_events,
        })
    return pd.DataFrame(rows)


# ── Per-cohort analysis ───────────────────────────────────────────────────────

def run_cohort_analysis(parquet_path: Path, cohort_label: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %s: %d rows × %d cols", cohort_label, len(df), df.shape[1])
    df = prepare_base_features(df)

    results = []

    for var, var_label in WELLNESS_VARS.items():
        if var not in df.columns:
            logger.warning("Variable %s not found in %s — skipping", var, cohort_label)
            continue

        # ── Unadjusted (univariate) ───────────────────────────────────────────
        univ = run_svy_logistic(
            df, [var], label=f"{cohort_label} | unadjusted | {var}"
        )
        row_univ = univ[univ["variable"] == var].iloc[0] if len(univ) > 0 else {}

        # ── Adjusted (added to 10-predictor base model) ───────────────────────
        adj_preds = BASE_PREDICTORS + [var]
        adj = run_svy_logistic(
            df, adj_preds, label=f"{cohort_label} | adjusted | {var}"
        )
        row_adj = adj[adj["variable"] == var].iloc[0] if len(adj) > 0 else {}

        # Weighted prevalence of this wellness variable
        mask = df[var].notna()
        w    = df.loc[mask, "WTMEC2YR"]
        prev = (df.loc[mask, var] * w).sum() / w.sum() * 100

        results.append({
            "variable":    var,
            "label":       var_label,
            "cohort":      cohort_label,
            "weighted_prev_pct": round(prev, 1),
            # Unadjusted
            "OR_univ":    row_univ.get("OR", np.nan),
            "CI_low_univ":row_univ.get("CI_low", np.nan),
            "CI_high_univ":row_univ.get("CI_high", np.nan),
            "p_univ":     row_univ.get("p_value", np.nan),
            # Adjusted
            "OR_adj":     row_adj.get("OR", np.nan),
            "CI_low_adj": row_adj.get("CI_low", np.nan),
            "CI_high_adj":row_adj.get("CI_high", np.nan),
            "p_adj":      row_adj.get("p_value", np.nan),
            "n_adj":      row_adj.get("n", np.nan),
        })

    return pd.DataFrame(results)


# ── Combined table (wide format) ──────────────────────────────────────────────

def make_combined_table(df_2017: pd.DataFrame, df_2021: pd.DataFrame) -> pd.DataFrame:
    m17 = df_2017.copy()
    m21 = df_2021.copy()

    cols_keep = [
        "variable", "label", "weighted_prev_pct",
        "OR_univ", "CI_low_univ", "CI_high_univ", "p_univ",
        "OR_adj", "CI_low_adj", "CI_high_adj", "p_adj",
    ]

    m17 = m17[cols_keep].rename(columns={
        c: f"{c}_2017" for c in cols_keep if c not in ("variable", "label")
    })
    m21 = m21[cols_keep].rename(columns={
        c: f"{c}_2021" for c in cols_keep if c not in ("variable", "label")
    })

    combined = m17.merge(m21, on=["variable", "label"], how="outer")
    combined["sig_adj_2017"] = (combined["p_adj_2017"] < 0.05).map({True: "✓", False: ""})
    combined["sig_adj_2021"] = (combined["p_adj_2021"] < 0.05).map({True: "✓", False: ""})
    return combined


# ── Side-by-side forest plot ──────────────────────────────────────────────────

def make_wellness_forest_plot(df_2017: pd.DataFrame, df_2021: pd.DataFrame, path: Path) -> None:
    labels   = list(WELLNESS_VARS.values())
    n_vars   = len(labels)
    x        = np.arange(n_vars)
    width    = 0.32

    def get_vals(df: pd.DataFrame, col: str) -> list:
        out = []
        for var in WELLNESS_VARS:
            row = df[df["variable"] == var]
            out.append(float(row[col].iloc[0]) if len(row) > 0 else np.nan)
        return out

    or_17   = get_vals(df_2017, "OR_adj")
    lo_17   = get_vals(df_2017, "CI_low_adj")
    hi_17   = get_vals(df_2017, "CI_high_adj")
    p_17    = get_vals(df_2017, "p_adj")

    or_21   = get_vals(df_2021, "OR_adj")
    lo_21   = get_vals(df_2021, "CI_low_adj")
    hi_21   = get_vals(df_2021, "CI_high_adj")
    p_21    = get_vals(df_2021, "p_adj")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    color_17 = "#4C72B0"
    color_21 = "#DD8452"

    for i in range(n_vars):
        for j, (or_v, lo, hi, pv, xpos, col) in enumerate([
            (or_17[i], lo_17[i], hi_17[i], p_17[i], x[i] - width/2, color_17),
            (or_21[i], lo_21[i], hi_21[i], p_21[i], x[i] + width/2, color_21),
        ]):
            if np.isnan(or_v):
                continue
            marker = "D" if pv < 0.05 else "o"
            ax.errorbar(
                xpos, or_v,
                yerr=[[or_v - lo], [hi - or_v]],
                fmt=marker, color=col, markersize=7, capsize=5,
                linewidth=1.8, elinewidth=1.8,
            )
            sig = "*" if pv < 0.05 else ""
            ax.text(xpos, hi + 0.04, f"{or_v:.2f}{sig}", ha="center", va="bottom",
                    fontsize=7.5, color=col, fontweight="bold" if pv < 0.05 else "normal")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Adjusted Odds Ratio (95% CI)", fontsize=9)
    ax.set_title(
        "Wellness Variable Adjusted ORs for Elevated ALT — Pre vs Post COVID\n"
        "(★ significant p<0.05 | adjusted for age, sex, PIR, waist, diabetes, smoking, triglycerides, metals)",
        fontsize=9, fontweight="bold",
    )

    p17_patch = mpatches.Patch(color=color_17, label="2017-2018 (pre-COVID)")
    p21_patch = mpatches.Patch(color=color_21, label="2021-2023 (post-COVID)")
    ax.legend(handles=[p17_patch, p21_patch], fontsize=9)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_17 = PROCESSED_DIR / "analytic_table.parquet"
    parquet_21 = PROCESSED_DIR / "analytic_table_2021.parquet"

    for p in (parquet_17, parquet_21):
        if not p.exists():
            logger.error("Missing: %s — run build scripts first", p)
            raise SystemExit(1)

    # ── Step B: 2017-2018 analysis ────────────────────────────────────────────
    logger.info("\n══════ 2017-2018 (Pre-COVID) Wellness Hypothesis ══════")
    df_2017 = run_cohort_analysis(parquet_17, "2017-2018")
    df_2017.to_csv(REPORTS_DIR / "wellness_hypothesis_2017.csv", index=False)
    logger.info("Saved reports/wellness_hypothesis_2017.csv")

    # ── Step C: 2021-2023 analysis ────────────────────────────────────────────
    logger.info("\n══════ 2021-2023 (Post-COVID) Wellness Hypothesis ══════")
    df_2021 = run_cohort_analysis(parquet_21, "2021-2023")
    df_2021.to_csv(REPORTS_DIR / "wellness_hypothesis_2021.csv", index=False)
    logger.info("Saved reports/wellness_hypothesis_2021.csv")

    # ── Step D: Combined comparison table ─────────────────────────────────────
    logger.info("\n══════ Combined Results ══════")
    combined = make_combined_table(df_2017, df_2021)
    combined.to_csv(REPORTS_DIR / "wellness_hypothesis_combined.csv", index=False)
    logger.info("Saved reports/wellness_hypothesis_combined.csv")

    # ── Forest plot ───────────────────────────────────────────────────────────
    make_wellness_forest_plot(df_2017, df_2021, FIGURES_DIR / "fig_wellness_forest_plot.png")

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n══════════════════════════════════════════════")
    logger.info("WELLNESS VARIABLE HYPOTHESIS RESULTS SUMMARY")
    logger.info("══════════════════════════════════════════════\n")

    for _, row in combined.iterrows():
        logger.info("  %-30s", row["label"])
        logger.info(
            "    2017-2018:  Prev=%.1f%%  OR_univ=%.3f (p=%.4f)  OR_adj=%.3f (p=%.4f) %s",
            row.get("weighted_prev_pct_2017", np.nan),
            row.get("OR_univ_2017", np.nan), row.get("p_univ_2017", np.nan),
            row.get("OR_adj_2017", np.nan),  row.get("p_adj_2017", np.nan),
            "✓ SIG" if row.get("p_adj_2017", 1) < 0.05 else "",
        )
        logger.info(
            "    2021-2023:  Prev=%.1f%%  OR_univ=%.3f (p=%.4f)  OR_adj=%.3f (p=%.4f) %s",
            row.get("weighted_prev_pct_2021", np.nan),
            row.get("OR_univ_2021", np.nan), row.get("p_univ_2021", np.nan),
            row.get("OR_adj_2021", np.nan),  row.get("p_adj_2021", np.nan),
            "✓ SIG" if row.get("p_adj_2021", 1) < 0.05 else "",
        )
        prev_change = row.get("weighted_prev_pct_2021", np.nan) - row.get("weighted_prev_pct_2017", np.nan)
        logger.info("    Prevalence change: %+.1f pp\n", prev_change)

    logger.info("Next step: run python src/14_prevalence_comparison.py")


if __name__ == "__main__":
    main()
