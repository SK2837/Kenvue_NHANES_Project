"""
Survey-weighted EDA for the NHANES 2017-2018 liver-injury screen.

Produces:
  reports/weighted_prevalence_table.csv
  figures/fig_alt_distribution.png
  figures/fig_bmi_by_alt.png
  figures/fig_correlation_heatmap.png
  figures/fig_prevalence_by_race.png
  figures/fig_metals_distribution.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase1_eda"
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase2_analysis"

plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-racial",
}


# ---------------------------------------------------------------------------
# Survey-weighted helpers
# ---------------------------------------------------------------------------

def weighted_prevalence(series: pd.Series, weights: pd.Series) -> float:
    """Horvitz-Thompson prevalence estimator: Σ(w·y) / Σw."""
    mask = series.notna() & weights.notna()
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna()
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


# ---------------------------------------------------------------------------
# Prevalence tables
# ---------------------------------------------------------------------------

def build_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    w = df["WTMEC2YR"]
    rows = []

    def add_row(group: str, category: str, mask: pd.Series) -> None:
        n = int(mask.sum())
        if n < 10:
            return
        p = weighted_prevalence(df.loc[mask, "ALT_elevated"], w[mask])
        rows.append({"Group": group, "Category": category,
                     "Weighted prevalence (%)": round(p * 100, 1), "N": n})

    # Overall
    add_row("Overall", "Overall", pd.Series(True, index=df.index))

    # Sex
    for code, label in [(1, "Male"), (2, "Female")]:
        add_row("Sex", label, df["RIAGENDR"] == code)

    # Age group
    bins   = [18, 40, 60, 120]
    labels = ["18-40", "41-60", "60+"]
    df["_age_grp"] = pd.cut(df["RIDAGEYR"], bins=bins, labels=labels, right=False)
    for lbl in labels:
        add_row("Age group", lbl, df["_age_grp"] == lbl)
    df.drop(columns=["_age_grp"], inplace=True)

    # Race / ethnicity
    for code, label in RACE_LABELS.items():
        add_row("Race/Ethnicity", label, df["RIDRETH3"] == code)

    # BMI category
    bmi_bins   = [0, 25, 30, 999]
    bmi_labels = ["<25 (Normal)", "25-30 (Overweight)", "≥30 (Obese)"]
    df["_bmi_cat"] = pd.cut(df["BMXBMI"], bins=bmi_bins, labels=bmi_labels)
    for lbl in bmi_labels:
        add_row("BMI category", lbl, df["_bmi_cat"] == lbl)
    df.drop(columns=["_bmi_cat"], inplace=True)

    # Diabetes
    for code, label in [(1, "Diabetes"), (2, "No diabetes"), (3, "Borderline")]:
        add_row("Diabetes status", label, df["DIQ010"] == code)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_alt_distribution(df: pd.DataFrame) -> None:
    alt = df["LBXSATSI"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(alt, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(33, color="salmon",  linestyle="--", label="Female threshold (33 U/L)")
    axes[0].axvline(56, color="coral",   linestyle="--", label="Male threshold (56 U/L)")
    axes[0].set_xlabel("ALT (U/L)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ALT Distribution (raw)")
    axes[0].legend()

    log_alt = np.log(alt[alt > 0])
    axes[1].hist(log_alt, bins=60, color="teal", edgecolor="white", alpha=0.8)
    axes[1].axvline(np.log(33), color="salmon", linestyle="--", label="Female threshold")
    axes[1].axvline(np.log(56), color="coral",  linestyle="--", label="Male threshold")
    axes[1].set_xlabel("ln(ALT)")
    axes[1].set_title("ALT Distribution (log-transformed)")
    axes[1].legend()

    plt.tight_layout()
    path = FIGURES_DIR / "fig_alt_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def fig_bmi_by_alt(df: pd.DataFrame) -> None:
    plot_df = df[["BMXBMI", "ALT_elevated"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))

    for val, label, color in [(0, "Normal ALT", "steelblue"), (1, "Elevated ALT", "salmon")]:
        subset = plot_df.loc[plot_df["ALT_elevated"] == val, "BMXBMI"]
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, density=True)

    ax.set_xlabel("BMI (kg/m²)")
    ax.set_ylabel("Density")
    ax.set_title("BMI Distribution by ALT Status")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "fig_bmi_by_alt.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    BIOMARKER_COLS = {
        "LBXSATSI": "ALT",
        "LBXSASSI": "AST",
        "LBXSAPSI": "ALP",
        "LBXSTB":   "Total Bilirubin",
        "LBXSAL":   "Albumin",
        "LBXBPB":   "Blood Lead",
        "LBXBCD":   "Blood Cadmium",
        "LBXTHG":   "Blood Mercury",
        "BMXBMI":   "BMI",
        "BMXWAIST": "Waist Circumference",
    }
    avail = [c for c in BIOMARKER_COLS if c in df.columns]
    corr_df = df[avail].rename(columns=BIOMARKER_COLS).corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Spearman Correlations — Continuous Biomarkers (analytic sample)")
    plt.tight_layout()
    path = FIGURES_DIR / "fig_correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def fig_prevalence_by_race(df: pd.DataFrame) -> None:
    w = df["WTMEC2YR"]
    race_prev = {}
    for code, label in RACE_LABELS.items():
        mask = df["RIDRETH3"] == code
        if mask.sum() >= 20:
            race_prev[label] = weighted_prevalence(df.loc[mask, "ALT_elevated"], w[mask])

    race_series = pd.Series(race_prev).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(race_series.index, race_series.values * 100,
                  color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Weighted prevalence of elevated ALT (%)")
    ax.set_title("ALT Elevation Prevalence by Race/Ethnicity (NHANES 2017–2018)")
    ax.set_xticks(range(len(race_series)))
    ax.set_xticklabels(race_series.index, rotation=20, ha="right")
    plt.tight_layout()
    path = FIGURES_DIR / "fig_prevalence_by_race.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def fig_metals_distribution(df: pd.DataFrame) -> None:
    metals = {
        "LBXBPB": "Blood Lead (µg/dL)",
        "LBXBCD": "Blood Cadmium (µg/L)",
        "LBXTHG": "Blood Mercury (µg/L)",
    }
    avail = {k: v for k, v in metals.items() if k in df.columns}
    if not avail:
        logger.warning("No metal columns found — skipping metals figure")
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(14, 4))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, avail.items()):
        vals = df[col].dropna()
        log_vals = np.log(vals[vals > 0])
        ax.hist(log_vals, bins=40, color="mediumpurple", edgecolor="white", alpha=0.8)
        ax.set_xlabel(f"ln({label})")
        ax.set_ylabel("Count")
        ax.set_title(f"ln({label.split()[1]}) distribution")

    plt.suptitle("Heavy Metal Distributions — log-transformed (analytic sample)", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_metals_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Spot-checks
# ---------------------------------------------------------------------------

def run_spot_checks(df: pd.DataFrame) -> None:
    w = df["WTMEC2YR"]

    bmi_mean = weighted_mean(df["BMXBMI"], w)
    logger.info("Weighted mean BMI: %.1f kg/m²  (expect 28–30)", bmi_mean)
    if not (26 <= bmi_mean <= 32):
        logger.warning("Weighted BMI %.1f outside expected range 26–32", bmi_mean)

    male_pct = weighted_prevalence((df["RIAGENDR"] == 1).astype(float), w)
    logger.info("Weighted %% male: %.1f%%  (expect 48–51%%)", male_pct * 100)
    if not (0.45 <= male_pct <= 0.55):
        logger.warning("%% male %.1f%% outside expected range 45–55%%", male_pct * 100)

    prev_overall = weighted_prevalence(df["ALT_elevated"], w)
    logger.info("Weighted ALT prevalence (overall): %.1f%%", prev_overall * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        logger.error("Analytic table not found — run 02_build_dataset.py first")
        raise SystemExit(1)

    df = pd.read_parquet(parquet)
    logger.info("Loaded analytic table: %s rows × %s cols", f"{len(df):,}", df.shape[1])

    # Spot-checks
    run_spot_checks(df)

    # Weighted prevalence table
    logger.info("Building weighted prevalence table ...")
    prev_table = build_prevalence_table(df)
    out_csv = REPORTS_DIR / "weighted_prevalence_table.csv"
    prev_table.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)
    logger.info("\n%s", prev_table.to_string(index=False))

    # Figures
    logger.info("Generating figures ...")
    fig_alt_distribution(df)
    fig_bmi_by_alt(df)
    fig_correlation_heatmap(df)
    fig_prevalence_by_race(df)
    fig_metals_distribution(df)

    logger.info("EDA complete.")


if __name__ == "__main__":
    main()
