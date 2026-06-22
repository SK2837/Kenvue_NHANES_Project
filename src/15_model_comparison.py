"""
Model Comparison: Same 10-Predictor Model on Both NHANES Cohorts
=================================================================
Fits the same survey-weighted logistic regression (10 predictors) to
both the 2017-2018 and 2021-2023 analytic tables and produces:
  - Side-by-side OR tables
  - Forest plot with both cohorts overlaid

Predictors (same as 04_inference.py):
  RIDAGEYR, sex_male, INDFMPIR, BMXWAIST, diabetes, ever_smoker,
  log_triglycerides, log_lead, log_cadmium, log_mercury

Outputs:
  reports/model_comparison_2017.csv
  reports/model_comparison_2021.csv
  reports/model_comparison_combined.csv
  figures/fig_model_comparison_forest.png
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

PREDICTORS = [
    "RIDAGEYR", "sex_male", "INDFMPIR", "BMXWAIST",
    "diabetes", "ever_smoker", "log_triglycerides",
    "log_lead", "log_cadmium", "log_mercury",
]

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

PREDICTOR_GROUPS = {
    "Demographics":    ["Age (years)", "Male sex", "Poverty-income ratio"],
    "Anthropometrics": ["Waist circumference (cm)"],
    "Clinical":        ["Diabetes", "Ever smoker", "Triglycerides (log)"],
    "Heavy metals":    ["log(Blood Lead)", "log(Blood Cadmium)", "log(Blood Mercury)"],
}

GROUP_COLOURS = {
    "Demographics":    "#4C72B0",
    "Anthropometrics": "#55A868",
    "Clinical":        "#C44E52",
    "Heavy metals":    "#8172B2",
}

OUTCOME = "ALT_elevated_40"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_lead"]          = np.log(out["LBXBPB"].clip(lower=0.001))
    out["log_cadmium"]       = np.log(out["LBXBCD"].clip(lower=0.001))
    out["log_mercury"]       = np.log(out["LBXTHG"].clip(lower=0.001))
    out["log_triglycerides"] = np.log(out["LBXSTR"].clip(lower=1))
    out["sex_male"]    = (out["RIAGENDR"] == 1).astype(float)
    out["diabetes"]    = (out["DIQ010"] == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan
    out["ever_smoker"] = (out["SMQ020"] == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan
    return out


def run_model(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    cols = [OUTCOME, "WTMEC2YR", "SDMVSTRA", "SDMVPSU"] + PREDICTORS
    model_df = df[cols].dropna()
    n        = len(model_df)
    n_events = int(model_df[OUTCOME].sum())
    epv      = n_events / len(PREDICTORS)
    logger.info("[%s] N=%d  events=%d  EPV=%.1f", cohort, n, n_events, epv)

    pl_df  = pl.from_pandas(model_df)
    design = svy.Design(stratum="SDMVSTRA", psu="SDMVPSU", wgt="WTMEC2YR")
    sample = svy.Sample(data=pl_df, design=design)
    fit    = sample.glm.fit(y=OUTCOME, x=PREDICTORS, family="binomial", link="logit")

    rows = []
    for coef in fit.coefs:
        if coef.term == "_intercept_":
            continue
        beta = float(coef.est)
        lo   = float(coef.lci)
        hi   = float(coef.uci)
        pval = float(coef.wald.p_value) if coef.wald is not None else np.nan
        label = COL_LABELS.get(coef.term, coef.term)
        rows.append({
            "variable": coef.term,
            "label":    label,
            "cohort":   cohort,
            "beta":     round(beta, 4),
            "OR":       round(np.exp(beta), 3),
            "CI_low":   round(np.exp(lo), 3),
            "CI_high":  round(np.exp(hi), 3),
            "p_value":  round(pval, 4),
            "significant": pval < 0.05,
            "n":        n,
            "n_events": n_events,
        })
    return pd.DataFrame(rows)


def make_forest_plot(df17: pd.DataFrame, df21: pd.DataFrame, path: Path) -> None:
    label_to_group = {
        lbl: grp
        for grp, lbls in PREDICTOR_GROUPS.items()
        for lbl in lbls
    }
    group_order = list(PREDICTOR_GROUPS.keys())

    # Merge and sort by group
    merged = df17[["variable", "label"]].copy()
    merged["group"] = merged["label"].map(label_to_group).fillna("Other")
    merged["group_rank"] = merged["group"].apply(
        lambda g: group_order.index(g) if g in group_order else len(group_order)
    )
    # Maintain within-group order from PREDICTORS list
    merged["pred_rank"] = merged["variable"].apply(
        lambda v: PREDICTORS.index(v) if v in PREDICTORS else 99
    )
    merged = merged.sort_values(["group_rank", "pred_rank"]).reset_index(drop=True)
    order = merged["variable"].tolist()

    n = len(order)
    y = np.arange(n)
    offset = 0.18
    color_17 = "#4C72B0"
    color_21 = "#DD8452"

    fig, ax = plt.subplots(figsize=(13, max(7, n * 0.65)))

    for i, var in enumerate(order):
        row17 = df17[df17["variable"] == var].iloc[0] if len(df17[df17["variable"] == var]) > 0 else None
        row21 = df21[df21["variable"] == var].iloc[0] if len(df21[df21["variable"] == var]) > 0 else None

        for row, ypos, col, label_suffix in [
            (row17, y[i] + offset, color_17, "2017"),
            (row21, y[i] - offset, color_21, "2021"),
        ]:
            if row is None:
                continue
            marker = "D" if row["p_value"] < 0.05 else "o"
            ax.errorbar(
                x=row["OR"], y=ypos,
                xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
                fmt=marker, color=col, markersize=7, capsize=4,
                linewidth=1.8, elinewidth=1.8, alpha=0.9,
            )
            sig = "*" if row["p_value"] < 0.05 else ""
            # Only annotate one side to avoid clutter
            if label_suffix == "2017":
                all_hi = max(
                    df17[df17["variable"] == var]["CI_high"].values[0] if len(df17[df17["variable"] == var]) > 0 else 0,
                    df21[df21["variable"] == var]["CI_high"].values[0] if len(df21[df21["variable"] == var]) > 0 else 0,
                )
                ax.text(
                    all_hi + 0.03, y[i],
                    (
                        f"17: {row17['OR']:.2f} ({row17['CI_low']:.2f}–{row17['CI_high']:.2f}){('*' if row17['p_value']<0.05 else '')}  "
                        f"21: {row21['OR']:.2f} ({row21['CI_low']:.2f}–{row21['CI_high']:.2f}){('*' if row21['p_value']<0.05 else '')}"
                    ) if row21 is not None else
                    f"17: {row17['OR']:.2f} ({row17['CI_low']:.2f}–{row17['CI_high']:.2f}){sig}",
                    va="center", fontsize=7.2, color="#333333",
                )

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"].tolist(), fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI) — survey-weighted, Taylor-series CIs", fontsize=9)
    ax.set_title(
        "10-Predictor Model: Pre-COVID (2017-2018) vs Post-COVID (2021-2023)\n"
        "(★ = p<0.05 | same model specification in both cohorts)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlim(left=0)

    group_patches = [
        mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOURS.items()
    ]
    cohort_patches = [
        mpatches.Patch(color=color_17, label="2017-2018 (pre-COVID)"),
        mpatches.Patch(color=color_21, label="2021-2023 (post-COVID)"),
    ]
    ax.legend(handles=group_patches + cohort_patches, fontsize=7.5, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_17 = PROCESSED_DIR / "analytic_table.parquet"
    parquet_21 = PROCESSED_DIR / "analytic_table_2021.parquet"
    for p in (parquet_17, parquet_21):
        if not p.exists():
            logger.error("Missing: %s", p)
            raise SystemExit(1)

    df17 = prepare_features(pd.read_parquet(parquet_17))
    df21 = prepare_features(pd.read_parquet(parquet_21))

    logger.info("\n── 2017-2018 model ──────────────────────────────────────────")
    or17 = run_model(df17, "2017-2018")
    or17.to_csv(REPORTS_DIR / "model_comparison_2017.csv", index=False)
    logger.info("\n%s", or17[["label", "OR", "CI_low", "CI_high", "p_value"]].to_string(index=False))

    logger.info("\n── 2021-2023 model ──────────────────────────────────────────")
    or21 = run_model(df21, "2021-2023")
    or21.to_csv(REPORTS_DIR / "model_comparison_2021.csv", index=False)
    logger.info("\n%s", or21[["label", "OR", "CI_low", "CI_high", "p_value"]].to_string(index=False))

    # Combined wide table
    combined = or17[["variable", "label", "OR", "CI_low", "CI_high", "p_value", "n", "n_events"]].rename(
        columns={"OR": "OR_2017", "CI_low": "CI_low_2017", "CI_high": "CI_high_2017",
                 "p_value": "p_2017", "n": "n_2017", "n_events": "events_2017"}
    ).merge(
        or21[["variable", "OR", "CI_low", "CI_high", "p_value", "n", "n_events"]].rename(
            columns={"OR": "OR_2021", "CI_low": "CI_low_2021", "CI_high": "CI_high_2021",
                     "p_value": "p_2021", "n": "n_2021", "n_events": "events_2021"}
        ),
        on="variable", how="outer",
    )
    combined["sig_2017"] = (combined["p_2017"] < 0.05).map({True: "✓", False: ""})
    combined["sig_2021"] = (combined["p_2021"] < 0.05).map({True: "✓", False: ""})
    combined["OR_change"] = (combined["OR_2021"] - combined["OR_2017"]).round(3)
    combined.to_csv(REPORTS_DIR / "model_comparison_combined.csv", index=False)
    logger.info("Saved reports/model_comparison_combined.csv")

    make_forest_plot(or17, or21, FIGURES_DIR / "fig_model_comparison_forest.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n══════ MODEL COMPARISON SUMMARY ══════\n")
    logger.info("%-30s  %8s  %8s  %10s  %8s  %8s  %10s",
                "Predictor", "OR(17)", "p(17)", "Sig(17)", "OR(21)", "p(21)", "Sig(21)")
    logger.info("-" * 80)
    for _, row in combined.iterrows():
        logger.info(
            "%-30s  %8.3f  %8.4f  %10s  %8.3f  %8.4f  %10s",
            row["label"],
            row.get("OR_2017", np.nan), row.get("p_2017", np.nan), row.get("sig_2017", ""),
            row.get("OR_2021", np.nan), row.get("p_2021", np.nan), row.get("sig_2021", ""),
        )

    n_sig_17 = (combined["p_2017"] < 0.05).sum()
    n_sig_21 = (combined["p_2021"] < 0.05).sum()
    logger.info("\nSignificant predictors (p<0.05): 2017-2018=%d  2021-2023=%d", n_sig_17, n_sig_21)
    logger.info("Next step: run python src/16_ml_comparison.py")


if __name__ == "__main__":
    main()
