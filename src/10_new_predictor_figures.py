"""
Generate EDA figures for the two newly discovered significant predictors:
  - Waist circumference by ALT status
  - Triglycerides by ALT status

Outputs:
  figures/fig_waist_by_alt.png
  figures/fig_triglycerides_by_alt.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase2_analysis"

NORMAL_COLOR   = "#4C72B0"   # steelblue — matches project palette
ELEVATED_COLOR = "#C44E52"   # salmon/red


def weighted_median_approx(values: pd.Series, weights: pd.Series) -> float:
    """Approximate weighted median via sorted cumulative weight."""
    df = pd.DataFrame({"v": values, "w": weights}).dropna().sort_values("v")
    cumw = df["w"].cumsum()
    half = df["w"].sum() / 2
    return float(df.loc[cumw >= half, "v"].iloc[0])


def annotate_significance(ax, p_value: float, or_value: float, label: str) -> None:
    """Add OR + significance annotation box in upper right."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    text = f"OR = {or_value:.2f}  (p = {p_value:.3f} {stars})\nFrom 10-predictor survey-weighted model"
    ax.text(
        0.97, 0.96, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="#ccaa00", alpha=0.9),
    )


def fig_waist_by_alt(df: pd.DataFrame) -> None:
    plot_df = df[["BMXWAIST", "ALT_elevated_40", "WTMEC2YR"]].dropna()
    w = plot_df["WTMEC2YR"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for val, label, color in [(0, "Normal ALT (≤40 U/L)", NORMAL_COLOR),
                               (1, "Elevated ALT (>40 U/L)", ELEVATED_COLOR)]:
        mask   = plot_df["ALT_elevated_40"] == val
        subset = plot_df.loc[mask, "BMXWAIST"]
        wt     = w[mask]

        ax.hist(subset, bins=45, alpha=0.55, label=label, color=color,
                density=True, edgecolor="none")

        # Weighted median line
        wmed = weighted_median_approx(subset, wt)
        ax.axvline(wmed, color=color, linewidth=1.8, linestyle="--", alpha=0.9)
        ax.text(wmed + 0.8, ax.get_ylim()[1] * 0.01,
                f"median\n{wmed:.0f} cm",
                color=color, fontsize=7.5, va="bottom")

    annotate_significance(ax, p_value=0.0115, or_value=1.027,
                          label="per 1 cm waist")

    ax.set_xlabel("Waist Circumference (cm)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Waist Circumference Distribution by ALT Status\n"
        "NHANES 2017–2018  |  Analytic sample (N = 3,543)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()

    path = FIGURES_DIR / "fig_waist_by_alt.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_triglycerides_by_alt(df: pd.DataFrame) -> None:
    if "LBXSTR" not in df.columns:
        print("LBXSTR (triglycerides) not found in parquet — skipping.")
        return

    df = df.copy()
    df["log_triglycerides"] = np.log(df["LBXSTR"].clip(lower=1))

    plot_df = df[["LBXSTR", "log_triglycerides", "ALT_elevated_40", "WTMEC2YR"]].dropna()
    w = plot_df["WTMEC2YR"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: raw triglycerides (mg/dL) ─────────────────────────────────
    ax = axes[0]
    for val, label, color in [(0, "Normal ALT (≤40 U/L)", NORMAL_COLOR),
                               (1, "Elevated ALT (>40 U/L)", ELEVATED_COLOR)]:
        mask   = plot_df["ALT_elevated_40"] == val
        subset = plot_df.loc[mask, "LBXSTR"]
        wt     = w[mask]

        ax.hist(subset, bins=60, alpha=0.55, label=label, color=color,
                density=True, edgecolor="none", range=(0, 600))

        wmed = weighted_median_approx(subset, wt)
        ax.axvline(wmed, color=color, linewidth=1.8, linestyle="--", alpha=0.9)
        # Offset label to avoid overlap: normal above, elevated below
        y_frac = 0.72 if val == 0 else 0.55
        ax.text(wmed + 4, ax.get_ylim()[1] * y_frac,
                f"median\n{wmed:.0f} mg/dL", color=color, fontsize=7.5,
                va="center", ha="left")

    ax.set_xlabel("Triglycerides (mg/dL)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Raw scale (right-skewed)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 600)

    # ── Right panel: log-transformed (model uses this scale) ──────────────────
    ax = axes[1]
    wmeds_log = {}
    for val, label, color in [(0, "Normal ALT (≤40 U/L)", NORMAL_COLOR),
                               (1, "Elevated ALT (>40 U/L)", ELEVATED_COLOR)]:
        mask   = plot_df["ALT_elevated_40"] == val
        subset = plot_df.loc[mask, "log_triglycerides"]
        wt     = w[mask]

        ax.hist(subset, bins=45, alpha=0.55, label=label, color=color,
                density=True, edgecolor="none")

        wmed = weighted_median_approx(subset, wt)
        wmeds_log[val] = wmed
        ax.axvline(wmed, color=color, linewidth=1.8, linestyle="--", alpha=0.9)

    # Offset median labels vertically to avoid overlap
    for val, color in [(0, NORMAL_COLOR), (1, ELEVATED_COLOR)]:
        wmed = wmeds_log[val]
        y_frac = 0.72 if val == 0 else 0.55
        ax.text(wmed + 0.05, ax.get_ylim()[1] * y_frac,
                f"median\n{wmed:.2f}", color=color, fontsize=7.5,
                va="center", ha="left")

    annotate_significance(ax, p_value=0.0179, or_value=1.773,
                          label="per log unit")

    # Add original mg/dL reference ticks on top x-axis
    ax2 = ax.twiny()
    log_ticks = np.log([50, 100, 150, 200, 300, 500])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(log_ticks)
    ax2.set_xticklabels(["50", "100", "150", "200", "300", "500"], fontsize=7)
    ax2.set_xlabel("Original mg/dL scale", fontsize=8, labelpad=2)

    ax.set_xlabel("log(Triglycerides)  — model scale", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Log-transformed (used in model)", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Triglycerides Distribution by ALT Status\n"
        "NHANES 2017–2018  |  Analytic sample (N = 3,543)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    path = FIGURES_DIR / "fig_triglycerides_by_alt.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        raise SystemExit("analytic_table.parquet not found — run 02_build_dataset.py first")

    df = pd.read_parquet(parquet)
    print(f"Loaded {len(df):,} rows")

    fig_waist_by_alt(df)
    fig_triglycerides_by_alt(df)
    print("Done.")


if __name__ == "__main__":
    main()
