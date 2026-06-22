"""
Pre/Post COVID Prevalence Comparison — NHANES 2017-2018 vs 2021-2023
=====================================================================
Produces survey-weighted prevalence estimates for key variables in both
cohorts, including ALT elevation rates, metabolic risk factors, and
COVID-era wellness variables, with subgroup breakdowns.

Outputs:
  reports/prevalence_comparison.csv   — full side-by-side table
  figures/fig_prevalence_comparison.png — bar chart comparison
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase3_covid"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase3_covid"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex_male"]   = (df["RIAGENDR"] == 1).astype(float)
    df["diabetes"]   = (df["DIQ010"] == 1).astype(float)
    df.loc[df["DIQ010"].isna(), "diabetes"] = np.nan
    df["ever_smoker"]= (df["SMQ020"] == 1).astype(float)
    df.loc[df["SMQ020"].isna(), "ever_smoker"] = np.nan
    df["obese"]      = (df["BMXBMI"] >= 30).astype(float)
    df.loc[df["BMXBMI"].isna(), "obese"] = np.nan
    df["age_18_39"]  = ((df["RIDAGEYR"] >= 18) & (df["RIDAGEYR"] < 40)).astype(float)
    df["age_40_59"]  = ((df["RIDAGEYR"] >= 40) & (df["RIDAGEYR"] < 60)).astype(float)
    df["age_60plus"] = (df["RIDAGEYR"] >= 60).astype(float)
    df["nh_white"]   = (df["RIDRETH3"] == 3).astype(float)
    df["nh_black"]   = (df["RIDRETH3"] == 4).astype(float)
    df["mex_am"]     = (df["RIDRETH3"] == 1).astype(float)
    df["nh_asian"]   = (df["RIDRETH3"] == 6).astype(float)
    # Waist obesity: NHANES thresholds (≥88cm women, ≥102cm men)
    waist_obese_m = (df["RIAGENDR"] == 1) & (df["BMXWAIST"] >= 102)
    waist_obese_f = (df["RIAGENDR"] == 2) & (df["BMXWAIST"] >= 88)
    df["central_obesity"] = (waist_obese_m | waist_obese_f).astype(float)
    df.loc[df["BMXWAIST"].isna(), "central_obesity"] = np.nan
    return df


def weighted_prev(df: pd.DataFrame, var: str) -> tuple[float, int]:
    mask = df[var].notna()
    if mask.sum() == 0:
        return np.nan, 0
    w   = df.loc[mask, "WTMEC2YR"]
    val = df.loc[mask, var]
    return float((val * w).sum() / w.sum() * 100), int(mask.sum())


def subgroup_prev(df: pd.DataFrame, var: str, group_col: str) -> dict[str, float]:
    out = {}
    for grp_val, grp_label in [(1, "male"), (2, "female")]:
        sub = df[df[group_col] == grp_val]
        prev, _ = weighted_prev(sub, var)
        out[grp_label] = prev
    return out


def build_prevalence_table(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []

    def add(label: str, var: str, group: str = "Overall", subgroup: str = "All"):
        prev, n = weighted_prev(df, var)
        rows.append({
            "group":    group,
            "subgroup": subgroup,
            "variable": var,
            "label":    label,
            "cohort":   cohort,
            "prev_pct": round(prev, 2) if not np.isnan(prev) else np.nan,
            "n":        n,
        })

    # ── ALT outcomes ──────────────────────────────────────────────────────────
    add("ALT > 40 U/L (all)",        "ALT_elevated_40",  "Primary Outcome")
    add("ALT elevated sex-specific",  "ALT_elevated",     "Primary Outcome")

    # By sex
    for sex_val, sex_lbl in [(1, "Male"), (2, "Female")]:
        sub = df[df["RIAGENDR"] == sex_val]
        prev, n = weighted_prev(sub, "ALT_elevated_40")
        rows.append({"group": "ALT by Sex", "subgroup": sex_lbl, "variable": "ALT_elevated_40",
                     "label": f"ALT>40 — {sex_lbl}", "cohort": cohort,
                     "prev_pct": round(prev, 2) if not np.isnan(prev) else np.nan, "n": n})

    # By age group
    for age_lbl, age_mask in [
        ("18-39", (df["RIDAGEYR"] < 40)),
        ("40-59", (df["RIDAGEYR"] >= 40) & (df["RIDAGEYR"] < 60)),
        ("60+",   (df["RIDAGEYR"] >= 60)),
    ]:
        sub = df[age_mask]
        prev, n = weighted_prev(sub, "ALT_elevated_40")
        rows.append({"group": "ALT by Age", "subgroup": age_lbl, "variable": "ALT_elevated_40",
                     "label": f"ALT>40 — {age_lbl}", "cohort": cohort,
                     "prev_pct": round(prev, 2) if not np.isnan(prev) else np.nan, "n": n})

    # By race/ethnicity
    for race_val, race_lbl in [
        (3, "NH White"), (4, "NH Black"), (1, "Mexican Am."), (6, "NH Asian"),
    ]:
        sub = df[df["RIDRETH3"] == race_val]
        prev, n = weighted_prev(sub, "ALT_elevated_40")
        rows.append({"group": "ALT by Race", "subgroup": race_lbl, "variable": "ALT_elevated_40",
                     "label": f"ALT>40 — {race_lbl}", "cohort": cohort,
                     "prev_pct": round(prev, 2) if not np.isnan(prev) else np.nan, "n": n})

    # ── Risk factors ──────────────────────────────────────────────────────────
    add("Diabetes",         "diabetes",       "Risk Factors")
    add("Obesity (BMI≥30)", "obese",          "Risk Factors")
    add("Central obesity",  "central_obesity","Risk Factors")
    add("Ever smoker",      "ever_smoker",    "Risk Factors")
    add("Male sex",         "sex_male",       "Demographics")

    # ── Wellness (COVID-era) variables ────────────────────────────────────────
    for var, lbl in [
        ("phq9_dep",      "Depression (PHQ-9 ≥10)"),
        ("short_sleep",   "Short sleep (<7h)"),
        ("high_sedentary","High sedentary (≥8h/day)"),
    ]:
        if var in df.columns:
            add(lbl, var, "Wellness / COVID-era")

    return pd.DataFrame(rows)


# ── Visualization ─────────────────────────────────────────────────────────────

def make_prevalence_figure(combined: pd.DataFrame, path: Path) -> None:
    # Focus on ALT outcomes and wellness variables
    plot_groups = ["Primary Outcome", "Wellness / COVID-era"]
    plot_df = combined[combined["group"].isin(plot_groups)].copy()

    labels_17 = {}
    labels_21 = {}
    for _, row in plot_df.iterrows():
        key = row["label"]
        if row["cohort"] == "2017-2018":
            labels_17[key] = row["prev_pct"]
        else:
            labels_21[key] = row["prev_pct"]

    all_labels = list(dict.fromkeys(plot_df["label"].tolist()))
    y = np.arange(len(all_labels))
    width = 0.35

    vals_17 = [labels_17.get(l, 0) for l in all_labels]
    vals_21 = [labels_21.get(l, 0) for l in all_labels]

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_labels) * 0.65)))
    b1 = ax.barh(y + width/2, vals_17, width, color="#4C72B0", label="2017-2018 (pre-COVID)", alpha=0.85)
    b2 = ax.barh(y - width/2, vals_21, width, color="#DD8452", label="2021-2023 (post-COVID)", alpha=0.85)

    for i, (v17, v21, lbl) in enumerate(zip(vals_17, vals_21, all_labels)):
        delta = v21 - v17
        sign  = "+" if delta >= 0 else ""
        ax.text(
            max(v17, v21) + 0.3, i,
            f"{sign}{delta:.1f}pp",
            va="center", fontsize=8,
            color="#006400" if abs(delta) < 1 else ("#B22222" if delta > 0 else "#006400"),
        )

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.set_xlabel("Weighted Prevalence (%)", fontsize=10)
    ax.set_title(
        "Prevalence Comparison: 2017-2018 vs 2021-2023\n(weighted, post-exclusion samples)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Risk factor comparison bar chart ──────────────────────────────────────────

def make_riskfactor_figure(combined: pd.DataFrame, path: Path) -> None:
    plot_groups = ["Risk Factors", "Demographics"]
    plot_df = combined[combined["group"].isin(plot_groups)].copy()

    labels_17 = {}
    labels_21 = {}
    for _, row in plot_df.iterrows():
        key = row["label"]
        if row["cohort"] == "2017-2018":
            labels_17[key] = row["prev_pct"]
        else:
            labels_21[key] = row["prev_pct"]

    all_labels = list(dict.fromkeys(plot_df["label"].tolist()))
    y = np.arange(len(all_labels))
    width = 0.35

    vals_17 = [labels_17.get(l, 0) for l in all_labels]
    vals_21 = [labels_21.get(l, 0) for l in all_labels]

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_labels) * 0.65)))
    ax.barh(y + width/2, vals_17, width, color="#4C72B0", label="2017-2018", alpha=0.85)
    ax.barh(y - width/2, vals_21, width, color="#DD8452", label="2021-2023", alpha=0.85)

    for i, (v17, v21) in enumerate(zip(vals_17, vals_21)):
        delta = v21 - v17
        sign  = "+" if delta >= 0 else ""
        ax.text(
            max(v17, v21) + 0.3, i,
            f"{sign}{delta:.1f}pp",
            va="center", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.set_xlabel("Weighted Prevalence (%)", fontsize=10)
    ax.set_title(
        "Risk Factor Prevalence: 2017-2018 vs 2021-2023",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()

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
            logger.error("Missing: %s", p)
            raise SystemExit(1)

    df17 = prepare_features(pd.read_parquet(parquet_17))
    df21 = prepare_features(pd.read_parquet(parquet_21))

    logger.info("2017-2018: %d rows | 2021-2023: %d rows", len(df17), len(df21))

    tbl17 = build_prevalence_table(df17, "2017-2018")
    tbl21 = build_prevalence_table(df21, "2021-2023")

    combined = pd.concat([tbl17, tbl21], ignore_index=True)
    combined.to_csv(REPORTS_DIR / "prevalence_comparison.csv", index=False)
    logger.info("Saved reports/prevalence_comparison.csv  (%d rows)", len(combined))

    make_prevalence_figure(combined, FIGURES_DIR / "fig_prevalence_comparison.png")
    make_riskfactor_figure(combined, FIGURES_DIR / "fig_riskfactor_comparison.png")

    # ── Print key numbers ─────────────────────────────────────────────────────
    logger.info("\n══════ KEY PREVALENCE COMPARISON ══════\n")
    key_vars = [
        "ALT > 40 U/L (all)", "Diabetes", "Obesity (BMI≥30)", "Central obesity",
        "Depression (PHQ-9 ≥10)", "Short sleep (<7h)", "High sedentary (≥8h/day)",
    ]
    pivot = combined[combined["label"].isin(key_vars)].pivot_table(
        index="label", columns="cohort", values="prev_pct"
    )
    if "2017-2018" in pivot.columns and "2021-2023" in pivot.columns:
        pivot["Change (pp)"] = (pivot["2021-2023"] - pivot["2017-2018"]).round(1)
        logger.info("\n%s\n", pivot.to_string())

    logger.info("Next step: run python src/15_model_comparison.py")


if __name__ == "__main__":
    main()
