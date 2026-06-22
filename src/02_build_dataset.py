"""
Merge NHANES 2017-2018 XPT modules, recode sentinels, define ALT outcome,
apply exclusion criteria, and save the analytic table to parquet.

Modules included:
  Core:    DEMO_J, BMX_J, BIOPRO_J, PBCD_J, ALQ_J, DIQ_J, SMQ_J, HEPB_S_J, HEPC_J
  Wellness (Phase 3 comparable): DPQ_J (PHQ-9), SLQ_J (sleep), PAQ_J (sedentary)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

MODULES = [
    "DEMO_J",
    "BMX_J",
    "BIOPRO_J",
    "PBCD_J",
    "ALQ_J",
    "DIQ_J",
    "SMQ_J",
    "HEPB_S_J",
    "HEPC_J",
    # Wellness / Phase 3 comparable modules
    "DPQ_J",  # PHQ-9 depression screener (DPQ010–DPQ090)
    "SLQ_J",  # Sleep hours (SLD012 weekday, SLD013 weekend)
    "PAQ_J",  # Physical activity / sedentary time (PAD680)
]

# ALQ121 frequency-of-drinking codes → estimated days per week
_FREQ_TO_DPW = {
    0: 0.0,    # Never
    1: 7.0,    # Every day
    2: 6.0,    # Nearly every day
    3: 3.5,    # 3-4 times/week
    4: 2.0,    # 2 times/week
    5: 1.0,    # Once a week
    6: 0.625,  # 2-3 times/month
    7: 0.25,   # Once a month
    8: 0.17,   # 7-11 times/year
    9: 0.087,  # 3-6 times/year
    10: 0.029, # 1-2 times/year
}

# PHQ-9 items (DPQ010–DPQ090); each scored 0–3
_PHQ9_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

# Non-response sentinels per variable (2017-2018 cycle)
SENTINELS: dict[str, list[int]] = {
    "ALQ121": [77, 99],
    "ALQ130": [777, 999],
    "DIQ010": [7, 9],
    "SMQ020": [7, 9],
    "SMQ040": [7, 9],
    "LBXHBS": [3],   # indeterminate → treat as missing
    "LBXHCR": [3],   # indeterminate → treat as missing
    # PHQ-9 items: 7=refused, 9=don't know
    **{item: [7, 9] for item in _PHQ9_ITEMS},
    # Sleep sentinels
    "SLD012": [99],
    "SLD013": [99],
    # Sedentary minutes sentinel
    "PAD680": [7777, 9999],
}


def load_xpt(module_code: str) -> pd.DataFrame:
    path = RAW_DIR / f"{module_code}.XPT"
    if not path.exists():
        logger.error("Missing XPT file: %s — run 01_download.py first", path)
        sys.exit(1)
    df, _ = pyreadstat.read_xport(str(path), encoding="latin1")
    logger.info("Loaded  %-12s  %s rows, %s cols", module_code, f"{len(df):,}", df.shape[1])
    return df


def merge_modules() -> pd.DataFrame:
    """Left-join all modules onto DEMO_J using SEQN.

    Drops columns already in the base frame before each merge to avoid
    duplicate-column conflicts from module-specific weight variables.
    """
    frames = {code: load_xpt(code) for code in MODULES}
    base = frames["DEMO_J"].copy()
    base_n = len(base)

    for code in MODULES[1:]:
        right = frames[code].copy()
        dup_cols = [c for c in right.columns if c in base.columns and c != "SEQN"]
        if dup_cols:
            right = right.drop(columns=dup_cols)
        base = base.merge(right, on="SEQN", how="left")
        assert len(base) == base_n, f"Row count changed after merging {code}"

    logger.info("Merged frame: %s rows × %s cols", f"{len(base):,}", base.shape[1])
    return base


def recode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    for col, codes in SENTINELS.items():
        if col not in df.columns:
            continue
        n = df[col].isin(codes).sum()
        if n:
            df[col] = df[col].replace(codes, np.nan)
            logger.info("Recoded  %-10s  %d sentinel values → NaN", col, n)
    return df


def engineer_wellness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PHQ-9, sleep, and sedentary features for Phase 3 comparability."""

    # PHQ-9 Depression Score
    phq_cols = [c for c in _PHQ9_ITEMS if c in df.columns]
    if phq_cols:
        phq_data = df[phq_cols]
        n_missing = phq_data.isna().sum(axis=1)
        df["phq9_score"] = phq_data.sum(axis=1, min_count=1)
        df.loc[n_missing > 3, "phq9_score"] = np.nan
        df["phq9_dep"] = (df["phq9_score"] >= 10).astype(float)
        df.loc[df["phq9_score"].isna(), "phq9_dep"] = np.nan
        logger.info(
            "PHQ-9: mean=%.1f  moderate dep (≥10)=%.1f%%",
            df["phq9_score"].mean(),
            df["phq9_dep"].mean() * 100,
        )

    # Sedentary behaviour (PAD680 = minutes/day sitting)
    if "PAD680" in df.columns:
        df["sedentary_hours"] = df["PAD680"] / 60.0
        df["high_sedentary"] = (df["sedentary_hours"] >= 8.0).astype(float)
        df.loc[df["sedentary_hours"].isna(), "high_sedentary"] = np.nan
        logger.info(
            "Sedentary: mean=%.1f hrs/day  high-sedentary (≥8h)=%.1f%%",
            df["sedentary_hours"].mean(),
            df["high_sedentary"].mean() * 100,
        )

    # Sleep (SLD012 = weekday hours)
    if "SLD012" in df.columns:
        df["sleep_hours"] = df["SLD012"]
        df["short_sleep"] = (df["sleep_hours"] < 7.0).astype(float)
        df.loc[df["sleep_hours"].isna(), "short_sleep"] = np.nan
        logger.info(
            "Sleep: mean=%.1f hrs/night  short sleep (<7h)=%.1f%%",
            df["sleep_hours"].mean(),
            df["short_sleep"].mean() * 100,
        )

    return df


def define_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    ALT_elevated: sex-specific AASLD thresholds (Kwo et al. 2017)
      male   > 56 U/L
      female > 33 U/L
    ALT_elevated_40: unisex 40 U/L sensitivity threshold
    """
    conditions = [
        (df["RIAGENDR"] == 1) & (df["LBXSATSI"] > 56),
        (df["RIAGENDR"] == 2) & (df["LBXSATSI"] > 33),
    ]
    df["ALT_elevated"] = np.select(conditions, [1, 1], default=0).astype(float)
    df.loc[df["LBXSATSI"].isna(), "ALT_elevated"] = np.nan

    df["ALT_elevated_40"] = (df["LBXSATSI"] > 40).astype(float)
    df.loc[df["LBXSATSI"].isna(), "ALT_elevated_40"] = np.nan

    prev = df["ALT_elevated"].mean()
    logger.info("Pre-exclusion unweighted ALT prevalence: %.1f%%", prev * 100)
    return df


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sequential exclusion criteria; return (analytic_df, flow_table_df)."""
    flow: list[dict] = []

    def exclude(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
        n_before = len(df)
        df = df[mask].copy()
        flow.append({"Step": label, "N before": n_before,
                     "N dropped": n_before - len(df), "N after": len(df)})
        return df

    df = exclude(df, df["RIDAGEYR"] >= 18,                        "1. Adults ≥18")
    df = exclude(df, df["WTMEC2YR"] > 0,                         "2. MEC-examined")
    df = exclude(df, df["LBXSATSI"].notna(),                     "3. ALT measured")
    df = exclude(df, df["LBXHBS"].ne(1) | df["LBXHBS"].isna(),  "4. HepB antigen −")
    df = exclude(df, df["LBXHCR"].ne(1) | df["LBXHCR"].isna(), "5. HepC antibody −")

    # Heavy alcohol exclusion (NIAAA): estimate weekly drinks from frequency × daily average
    df["_freq_dpw"] = df["ALQ121"].map(_FREQ_TO_DPW)
    df["_weekly_drinks"] = df["_freq_dpw"] * df["ALQ130"]
    male_heavy   = (df["RIAGENDR"] == 1) & (df["_weekly_drinks"] > 14)
    female_heavy = (df["RIAGENDR"] == 2) & (df["_weekly_drinks"] > 7)
    df = exclude(df, ~(male_heavy | female_heavy), "6. Non-excessive alcohol")
    df.drop(columns=["_freq_dpw", "_weekly_drinks"], inplace=True)

    return df, pd.DataFrame(flow)


def save_codebook(df: pd.DataFrame) -> None:
    """Save a plain-English codebook for all columns in the analytic table."""
    LABELS: dict[str, tuple[str, str]] = {
        "SEQN":           ("DEMO_J",    "Unique participant identifier"),
        "RIDAGEYR":       ("DEMO_J",    "Age in years at screening"),
        "RIAGENDR":       ("DEMO_J",    "Sex (1=Male, 2=Female)"),
        "RIDRETH3":       ("DEMO_J",    "Race/Ethnicity (1=Mexican American, 3=NHWhite, 4=NHBlack, 6=NHAsian)"),
        "INDFMPIR":       ("DEMO_J",    "Family poverty-income ratio"),
        "WTMEC2YR":       ("DEMO_J",    "2-year MEC exam sample weight"),
        "SDMVSTRA":       ("DEMO_J",    "Masked variance pseudo-stratum"),
        "SDMVPSU":        ("DEMO_J",    "Masked variance pseudo-PSU"),
        "BMXBMI":         ("BMX_J",     "Body mass index (kg/m²)"),
        "BMXWAIST":       ("BMX_J",     "Waist circumference (cm)"),
        "LBXSATSI":       ("BIOPRO_J",  "Alanine aminotransferase / ALT (U/L)"),
        "LBXSASSI":       ("BIOPRO_J",  "Aspartate aminotransferase / AST (U/L)"),
        "LBXSAPSI":       ("BIOPRO_J",  "Alkaline phosphatase / ALP (U/L)"),
        "LBXSTB":         ("BIOPRO_J",  "Total bilirubin (mg/dL)"),
        "LBXSAL":         ("BIOPRO_J",  "Albumin (g/dL)"),
        "LBXSGL":         ("BIOPRO_J",  "Glucose (mg/dL)"),
        "LBXSTR":         ("BIOPRO_J",  "Triglycerides (mg/dL)"),
        "LBXBPB":         ("PBCD_J",    "Blood lead (µg/dL)"),
        "LBXBCD":         ("PBCD_J",    "Blood cadmium (µg/L)"),
        "LBXTHG":         ("PBCD_J",    "Blood mercury (µg/L)"),
        "ALQ121":         ("ALQ_J",     "Drinking frequency past 12 months (0=Never…1=Every day)"),
        "ALQ130":         ("ALQ_J",     "Average drinks per day on drinking days"),
        "DIQ010":         ("DIQ_J",     "Doctor-diagnosed diabetes (1=Yes, 2=No, 3=Borderline)"),
        "SMQ020":         ("SMQ_J",     "Smoked ≥100 cigarettes in lifetime (1=Yes, 2=No)"),
        "SMQ040":         ("SMQ_J",     "Currently smoking (1=Every day, 2=Some days, 3=Not at all)"),
        "LBXHBS":         ("HEPB_S_J", "Hepatitis B surface antigen (1=Positive, 2=Negative)"),
        "LBXHCR":         ("HEPC_J",   "Hepatitis C antibody (1=Positive, 2=Negative)"),
        "SLD012":         ("SLQ_J",    "Weekday sleep hours"),
        "SLD013":         ("SLQ_J",    "Weekend sleep hours"),
        "PAD680":         ("PAQ_J",    "Sedentary activity minutes per day"),
        "ALT_elevated":    ("derived", "Primary outcome: elevated ALT by sex-specific AASLD threshold"),
        "ALT_elevated_40": ("derived", "Sensitivity outcome: ALT > 40 U/L (unisex)"),
        "phq9_score":      ("derived", "PHQ-9 depression score (0–27; sum DPQ010–DPQ090)"),
        "phq9_dep":        ("derived", "Moderate-severe depression flag (PHQ-9 ≥ 10)"),
        "sedentary_hours": ("derived", "Sedentary hours per day (PAD680 / 60)"),
        "high_sedentary":  ("derived", "High sedentary flag (≥ 8 hours/day)"),
        "sleep_hours":     ("derived", "Weekday sleep hours (= SLD012)"),
        "short_sleep":     ("derived", "Short sleep flag (< 7 hours/night)"),
    }
    # Add PHQ-9 item labels
    for item in _PHQ9_ITEMS:
        LABELS[item] = ("DPQ_J", f"PHQ-9 item {item} (0=Not at all, 3=Nearly every day)")

    rows = []
    for col in df.columns:
        src, label = LABELS.get(col, ("", ""))
        rows.append({"variable": col, "source_module": src, "label": label})
    out = PROCESSED_DIR / "codebook.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Codebook saved → %s", out)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = merge_modules()
    df = recode_sentinels(df)
    df = engineer_wellness_features(df)
    df = define_outcome(df)
    df, flow_df = apply_exclusions(df)

    logger.info("\nExclusion flow:\n%s", flow_df.to_string(index=False))
    logger.info("Final analytic N: %s", f"{len(df):,}")

    # Sanity checks
    assert (df["WTMEC2YR"] > 0).all(), "Negative/zero weights in analytic sample"
    assert df["LBXSATSI"].notna().all(), "Missing ALT values remain after exclusion step 3"
    assert df["ALT_elevated"].notna().all(), "NaN in ALT_elevated outcome"

    # Weighted prevalence of wellness variables post-exclusion
    w = df["WTMEC2YR"]
    if "phq9_dep" in df.columns:
        dep_prev = (df["phq9_dep"].fillna(0) * w).sum() / w[df["phq9_dep"].notna()].sum()
        logger.info("Weighted depression (PHQ-9 ≥10): %.1f%%", dep_prev * 100)
    if "high_sedentary" in df.columns:
        sed_prev = (df["high_sedentary"].fillna(0) * w).sum() / w[df["high_sedentary"].notna()].sum()
        logger.info("Weighted high sedentary (≥8h): %.1f%%", sed_prev * 100)
    if "short_sleep" in df.columns:
        sleep_prev = (df["short_sleep"].fillna(0) * w).sum() / w[df["short_sleep"].notna()].sum()
        logger.info("Weighted short sleep (<7h): %.1f%%", sleep_prev * 100)

    out = PROCESSED_DIR / "analytic_table.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved analytic table → %s  (%s rows × %s cols)", out, f"{len(df):,}", df.shape[1])

    save_codebook(df)

    # Save exclusion flow
    flow_path = PROCESSED_DIR / "exclusion_flow_2017.csv"
    flow_df.to_csv(flow_path, index=False)
    logger.info("Exclusion flow → %s", flow_path)


if __name__ == "__main__":
    main()
