"""
Export the analytic table to human-readable CSVs for inspection.

Outputs:
  data/processed/inspect_full.csv          — all 3,543 rows, all columns
  data/processed/inspect_key_columns.csv   — only the 25 most meaningful columns, human-labeled
  data/processed/inspect_summary_stats.csv — mean / median / min / max / missing % per key column
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Human-readable names for the key columns we care about
KEY_COLS = {
    "SEQN":         "Participant ID",
    "RIDAGEYR":     "Age (years)",
    "RIAGENDR":     "Sex (1=Male 2=Female)",
    "RIDRETH3":     "Race/Ethnicity",
    "INDFMPIR":     "Poverty-Income Ratio",
    "WTMEC2YR":     "Survey Weight",
    "SDMVSTRA":     "Survey Stratum",
    "SDMVPSU":      "Survey PSU",
    "BMXBMI":       "BMI (kg/m²)",
    "BMXWAIST":     "Waist Circumference (cm)",
    "LBXSATSI":     "ALT (U/L)  ← outcome variable",
    "LBXSASSI":     "AST (U/L)",
    "LBXSAPSI":     "ALP (U/L)",
    "LBXSTB":       "Total Bilirubin (mg/dL)",
    "LBXSAL":       "Albumin (g/dL)",
    "LBXSGL":       "Glucose (mg/dL)",
    "LBXBPB":       "Blood Lead (µg/dL)",
    "LBXBCD":       "Blood Cadmium (µg/L)",
    "LBXTHG":       "Blood Mercury (µg/L)",
    "ALQ121":       "Drinking Frequency (0=Never…1=Every day)",
    "ALQ130":       "Avg Drinks/Day",
    "DIQ010":       "Diabetes Diagnosis (1=Yes 2=No 3=Borderline)",
    "SMQ020":       "Ever Smoked 100+ Cigarettes (1=Yes 2=No)",
    "ALT_elevated": "Elevated ALT — PRIMARY OUTCOME (1=Yes 0=No)",
    "ALT_elevated_40": "Elevated ALT >40 U/L — sensitivity outcome",
}

RACE_MAP = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
            4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/Multi-racial"}
SEX_MAP  = {1: "Male", 2: "Female"}
DIA_MAP  = {1: "Diabetes", 2: "No diabetes", 3: "Borderline"}
SMQ_MAP  = {1: "Yes", 2: "No"}


def main() -> None:
    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        raise FileNotFoundError("Run 02_build_dataset.py first")

    df = pd.read_parquet(parquet)
    print(f"Analytic table: {len(df):,} rows × {df.shape[1]} cols\n")

    # ── 1. Full export ──────────────────────────────────────────────────────
    full_path = PROCESSED_DIR / "inspect_full.csv"
    df.to_csv(full_path, index=False)
    print(f"Full CSV saved  →  {full_path}")

    # ── 2. Key columns only, with decoded labels ────────────────────────────
    avail = [c for c in KEY_COLS if c in df.columns]
    key_df = df[avail].copy()

    # Decode categorical codes to readable strings
    key_df["RIAGENDR"] = key_df["RIAGENDR"].map(SEX_MAP)
    key_df["RIDRETH3"] = key_df["RIDRETH3"].map(RACE_MAP)
    key_df["DIQ010"]   = key_df["DIQ010"].map(DIA_MAP)
    key_df["SMQ020"]   = key_df["SMQ020"].map(SMQ_MAP)

    # Rename columns to human-readable labels
    key_df.rename(columns=KEY_COLS, inplace=True)

    key_path = PROCESSED_DIR / "inspect_key_columns.csv"
    key_df.to_csv(key_path, index=False)
    print(f"Key columns CSV →  {key_path}")

    # ── 3. Summary statistics ───────────────────────────────────────────────
    numeric_cols = [c for c in avail if df[c].dtype in [float, np.float64, np.float32,
                                                          int, np.int64, np.int32]
                    and c not in ("SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR")]

    stats_rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        stats_rows.append({
            "Variable":    col,
            "Label":       KEY_COLS.get(col, ""),
            "N non-null":  len(s),
            "Missing %":   round(df[col].isna().mean() * 100, 1),
            "Mean":        round(s.mean(), 3),
            "Median":      round(s.median(), 3),
            "Std":         round(s.std(), 3),
            "Min":         round(s.min(), 3),
            "Max":         round(s.max(), 3),
            "% = 1 (if binary)": round((s == 1).mean() * 100, 1) if s.isin([0, 1]).all() else "",
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = PROCESSED_DIR / "inspect_summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Summary stats   →  {stats_path}")

    # ── Print a quick preview ───────────────────────────────────────────────
    print("\n── First 5 rows (key columns) ──────────────────────────────────")
    print(key_df.head().to_string(index=False))

    print("\n── Summary statistics (numeric key variables) ──────────────────")
    print(stats_df[["Variable", "Label", "N non-null", "Missing %",
                     "Mean", "Median", "Min", "Max"]].to_string(index=False))

    print("\n── ALT outcome distribution ────────────────────────────────────")
    alt_counts = df["ALT_elevated"].value_counts().sort_index()
    for val, count in alt_counts.items():
        label = "Elevated" if val == 1 else "Normal"
        print(f"  {label} ({int(val)}): {count:,}  ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
