"""
Merge NHANES 2021-2023 XPT modules, apply the same exclusion criteria as the
2017-2018 pipeline, engineer features, and add COVID-era variables.

Key differences vs 02_build_dataset.py:
  - Source: data/raw_2021/ (_L suffix files)
  - Output: data/processed/analytic_table_2021.parquet
  - ALQ gating: ALQ111=2 (never drank) → weekly_drinks=0 (2021 structure)
  - New features: PHQ-9 depression score, sedentary hours, sleep hours

Outputs:
  data/processed/analytic_table_2021.parquet
  data/processed/codebook_2021.csv
  data/processed/exclusion_flow_2021.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw_2021"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

MODULES = [
    "DEMO_L", "BMX_L", "BIOPRO_L", "PBCD_L",
    "ALQ_L", "DIQ_L", "SMQ_L",
    "HEPB_S_L", "HEPC_L",
    "DPQ_L", "PAQ_L", "SLQ_L",
]

# ALQ121 frequency codes → estimated days per week (same as 2017-2018)
_FREQ_TO_DPW = {
    0: 0.0, 1: 7.0, 2: 6.0, 3: 3.5, 4: 2.0, 5: 1.0,
    6: 0.625, 7: 0.25, 8: 0.17, 9: 0.087, 10: 0.029,
}

# PHQ-9 items (DPQ010–DPQ090); each scored 0–3
_PHQ9_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

SENTINELS: dict[str, list] = {
    "ALQ121": [77, 99],
    "ALQ130": [777, 999],
    "DIQ010": [7, 9],
    "SMQ020": [7, 9],
    "LBXHBS": [3],
    "LBXHCR": [3],
    # PHQ-9: 7=refused, 9=don't know
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
        logger.error("Missing: %s — run 11_download_nhanes_2021.py first", path)
        sys.exit(1)
    df, _ = pyreadstat.read_xport(str(path), encoding="latin1")
    logger.info("Loaded  %-12s  %s rows × %s cols", module_code, f"{len(df):,}", df.shape[1])
    return df


def merge_modules() -> pd.DataFrame:
    """Left-join all modules onto DEMO_L using SEQN.

    Drops columns that already exist in the base frame before each merge to
    avoid duplicate column conflicts from module-specific weight variables
    (e.g. WTPH2YR appears in multiple 2021-2023 modules).
    """
    frames = {code: load_xpt(code) for code in MODULES}
    base   = frames["DEMO_L"].copy()
    n      = len(base)
    for code in MODULES[1:]:
        right = frames[code].copy()
        # Drop columns already in base except SEQN (the join key)
        dup_cols = [c for c in right.columns if c in base.columns and c != "SEQN"]
        if dup_cols:
            right = right.drop(columns=dup_cols)
        base = base.merge(right, on="SEQN", how="left")
        assert len(base) == n, f"Row count changed merging {code}"
    logger.info("Merged: %s rows × %s cols", f"{len(base):,}", base.shape[1])
    return base


def recode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    for col, codes in SENTINELS.items():
        if col not in df.columns:
            continue
        n = df[col].isin(codes).sum()
        if n:
            df[col] = df[col].replace(codes, np.nan)
            logger.info("Recoded  %-10s  %d sentinels → NaN", col, n)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns, including COVID-era variables."""

    # ── Standard features (same as 2017-2018) ─────────────────────────────────
    df["log_lead"]    = np.log(df["LBXBPB"].clip(lower=0.01))
    df["log_cadmium"] = np.log(df["LBXBCD"].clip(lower=0.01))
    df["log_mercury"] = np.log(df["LBXTHG"].clip(lower=0.01))
    df["sex_male"]    = (df["RIAGENDR"] == 1).astype(float)

    df["race_mex_am"]    = (df["RIDRETH3"] == 1).astype(float)
    df["race_oth_hisp"]  = (df["RIDRETH3"] == 2).astype(float)
    df["race_nh_black"]  = (df["RIDRETH3"] == 4).astype(float)
    df["race_nh_asian"]  = (df["RIDRETH3"] == 6).astype(float)
    df["race_oth_multi"] = (df["RIDRETH3"] == 7).astype(float)

    df["diabetes"]    = (df["DIQ010"] == 1).astype(float)
    df.loc[df["DIQ010"].isna(), "diabetes"] = np.nan

    df["ever_smoker"] = (df["SMQ020"] == 1).astype(float)
    df.loc[df["SMQ020"].isna(), "ever_smoker"] = np.nan

    df["log_triglycerides"] = np.log(df["LBXSTR"].clip(lower=1))

    # ── Alcohol — handle 2021-2023 gating structure ───────────────────────────
    # ALQ111=2 means "never drank ≥12 drinks in life" → set weekly drinks to 0
    # ALQ111=1 (ever drank) → compute from ALQ121 × ALQ130
    df["_freq_dpw"]     = df["ALQ121"].map(_FREQ_TO_DPW)
    df["_weekly_drinks"] = df["_freq_dpw"] * df["ALQ130"]
    # Non-drinkers: ALQ111=2 → their ALQ121/130 are NaN; set to 0
    never_drinker = df["ALQ111"] == 2
    df.loc[never_drinker, "_weekly_drinks"] = 0.0
    df.loc[never_drinker, "_freq_dpw"]      = 0.0

    df["log_weekly_drinks"] = np.log1p(df["_weekly_drinks"])

    # ── PHQ-9 Depression Score (new COVID-era variable) ────────────────────────
    # Score = sum of DPQ010–DPQ090 (each 0–3); range 0–27
    # Participants with >3 missing items get NaN (unreliable total)
    phq_cols_present = [c for c in _PHQ9_ITEMS if c in df.columns]
    if phq_cols_present:
        phq_data        = df[phq_cols_present]
        n_missing       = phq_data.isna().sum(axis=1)
        df["phq9_score"] = phq_data.sum(axis=1, min_count=1)
        df.loc[n_missing > 3, "phq9_score"] = np.nan
        # Binary: moderate-severe depression (≥10)
        df["phq9_dep"] = (df["phq9_score"] >= 10).astype(float)
        df.loc[df["phq9_score"].isna(), "phq9_dep"] = np.nan
        logger.info("PHQ-9 score: mean=%.1f  moderate dep (≥10)=%.1f%%",
                    df["phq9_score"].mean(),
                    df["phq9_dep"].mean() * 100)

    # ── Sedentary behaviour (new COVID-era variable) ───────────────────────────
    # PAD680 = minutes of sedentary activity per day
    if "PAD680" in df.columns:
        df["sedentary_hours"] = df["PAD680"] / 60.0
        # Binary: highly sedentary = ≥8 hours/day (480 min)
        df["high_sedentary"] = (df["sedentary_hours"] >= 8.0).astype(float)
        df.loc[df["sedentary_hours"].isna(), "high_sedentary"] = np.nan
        logger.info("Sedentary: mean=%.1f hrs/day  high-sedentary (≥8h)=%.1f%%",
                    df["sedentary_hours"].mean(),
                    df["high_sedentary"].mean() * 100)

    # ── Sleep (new COVID-era variable) ─────────────────────────────────────────
    # SLD012 = weekday sleep hours; SLD013 = weekend sleep hours
    if "SLD012" in df.columns:
        df["sleep_hours"] = df["SLD012"]
        # Binary: short sleep = <7 hours (CDC recommendation)
        df["short_sleep"] = (df["sleep_hours"] < 7.0).astype(float)
        df.loc[df["sleep_hours"].isna(), "short_sleep"] = np.nan
        logger.info("Sleep: mean=%.1f hrs/night  short sleep (<7h)=%.1f%%",
                    df["sleep_hours"].mean(),
                    df["short_sleep"].mean() * 100)

    return df


def define_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Same ALT outcome definitions as 2017-2018."""
    conditions = [
        (df["RIAGENDR"] == 1) & (df["LBXSATSI"] > 56),
        (df["RIAGENDR"] == 2) & (df["LBXSATSI"] > 33),
    ]
    df["ALT_elevated"] = np.select(conditions, [1, 1], default=0).astype(float)
    df.loc[df["LBXSATSI"].isna(), "ALT_elevated"] = np.nan

    df["ALT_elevated_40"] = (df["LBXSATSI"] > 40).astype(float)
    df.loc[df["LBXSATSI"].isna(), "ALT_elevated_40"] = np.nan

    prev = df["ALT_elevated_40"].mean()
    logger.info("Pre-exclusion unweighted ALT>40 prevalence: %.1f%%", prev * 100)
    return df


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identical exclusion logic to 2017-2018 pipeline."""
    flow: list[dict] = []

    def exclude(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
        n_before = len(df)
        df = df[mask].copy()
        flow.append({"Step": label, "N_before": n_before,
                     "N_dropped": n_before - len(df), "N_after": len(df)})
        return df

    df = exclude(df, df["RIDAGEYR"] >= 18,                        "1. Adults ≥18")
    df = exclude(df, df["WTMEC2YR"] > 0,                         "2. MEC-examined")
    df = exclude(df, df["LBXSATSI"].notna(),                     "3. ALT measured")
    df = exclude(df, df["LBXHBS"].ne(1) | df["LBXHBS"].isna(),  "4. HepB antigen −")
    df = exclude(df, df["LBXHCR"].ne(1) | df["LBXHCR"].isna(), "5. HepC antibody −")

    # Heavy alcohol exclusion (>14 drinks/week men, >7 women)
    male_heavy   = (df["RIAGENDR"] == 1) & (df["_weekly_drinks"] > 14)
    female_heavy = (df["RIAGENDR"] == 2) & (df["_weekly_drinks"] > 7)
    df = exclude(df, ~(male_heavy | female_heavy), "6. Non-excessive alcohol")

    df.drop(columns=["_freq_dpw", "_weekly_drinks"], inplace=True)
    return df, pd.DataFrame(flow)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = merge_modules()
    df = recode_sentinels(df)
    df = engineer_features(df)
    df = define_outcome(df)
    df, flow_df = apply_exclusions(df)

    logger.info("\nExclusion flow:\n%s", flow_df.to_string(index=False))
    logger.info("Final analytic N: %s", f"{len(df):,}")

    # Sanity checks
    assert (df["WTMEC2YR"] > 0).all(), "Negative/zero weights"
    assert df["LBXSATSI"].notna().all(), "Missing ALT after exclusion step 3"

    # Weighted prevalence
    w = df["WTMEC2YR"]
    alt40_prev = (df["ALT_elevated_40"] * w).sum() / w.sum()
    logger.info("Weighted ALT>40 prevalence (post-exclusion): %.2f%%", alt40_prev * 100)

    out = PROCESSED_DIR / "analytic_table_2021.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved → %s  (%s rows × %s cols)", out, f"{len(df):,}", df.shape[1])

    # Save exclusion flow
    flow_path = PROCESSED_DIR / "exclusion_flow_2021.csv"
    flow_df.to_csv(flow_path, index=False)
    logger.info("Exclusion flow → %s", flow_path)

    # Quick comparison log
    logger.info("\n── Quick comparison with 2017-2018 ──────────────────────────")
    orig_path = PROCESSED_DIR / "analytic_table.parquet"
    if orig_path.exists():
        orig = pd.read_parquet(orig_path)
        w0 = orig["WTMEC2YR"]
        prev_2017 = (orig["ALT_elevated_40"] * w0).sum() / w0.sum() * 100
        prev_2021 = alt40_prev * 100
        logger.info("  2017-2018  N=%s  weighted ALT>40=%.2f%%", f"{len(orig):,}", prev_2017)
        logger.info("  2021-2023  N=%s  weighted ALT>40=%.2f%%", f"{len(df):,}", prev_2021)
        logger.info("  Change: %+.2f pp", prev_2021 - prev_2017)

    logger.info("Next step: run  python src/13_prevalence_comparison.py")


if __name__ == "__main__":
    main()
