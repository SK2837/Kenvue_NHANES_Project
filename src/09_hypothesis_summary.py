"""
Generate a comprehensive hypothesis test summary CSV combining results from:
  1. Original 9-predictor model (BMI-based)
  2. Updated 10-predictor model (waist + triglycerides replacing BMI)
  3. Extended hypothesis test (additional social/clinical variables)

Output: reports/hypothesis_test_all_variables.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "phase2_analysis"


def load_table(fname: str) -> pd.DataFrame:
    path = REPORTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"{fname} not found in {REPORTS_DIR}")
    return pd.read_csv(path)


def sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    orig   = load_table("inference_OR_table_original_9pred.csv")
    updated = load_table("inference_OR_table.csv")
    ext    = load_table("extended_OR_table.csv")

    # ── Category mapping for all variables ──────────────────────────────────────
    CATEGORY = {
        "RIDAGEYR":          "Demographics",
        "sex_male":          "Demographics",
        "INDFMPIR":          "Demographics",
        "BMXBMI":            "Anthropometrics",
        "BMXWAIST":          "Anthropometrics",
        "diabetes":          "Clinical",
        "ever_smoker":       "Behavioral",
        "log_triglycerides": "Clinical",
        "log_weekly_drinks": "Behavioral",
        "college_edu":       "Social",
        "log_lead":          "Environmental",
        "log_cadmium":       "Environmental",
        "log_mercury":       "Environmental",
    }

    LABEL = {
        "RIDAGEYR":          "Age (years)",
        "sex_male":          "Male sex",
        "INDFMPIR":          "Poverty-income ratio",
        "BMXBMI":            "BMI (kg/m²)",
        "BMXWAIST":          "Waist circumference (cm)",
        "diabetes":          "Diabetes",
        "ever_smoker":       "Ever smoker",
        "log_triglycerides": "Triglycerides (log)",
        "log_weekly_drinks": "Weekly alcohol drinks (log)",
        "college_edu":       "College education",
        "log_lead":          "log(Blood Lead)",
        "log_cadmium":       "log(Blood Cadmium)",
        "log_mercury":       "log(Blood Mercury)",
    }

    # All variables tested across any model
    ALL_VARS = [
        "RIDAGEYR", "sex_male", "INDFMPIR",
        "BMXBMI", "BMXWAIST",
        "diabetes", "ever_smoker",
        "log_triglycerides", "log_weekly_drinks", "college_edu",
        "log_lead", "log_cadmium", "log_mercury",
    ]

    def lookup(df: pd.DataFrame, var: str, col: str):
        row = df[df["variable"] == var]
        if row.empty or col not in row.columns:
            return np.nan
        val = row[col].iloc[0]
        return val if not pd.isna(val) else np.nan

    rows = []
    for var in ALL_VARS:
        # Original model (9-predictor, BMI-based)
        o_or   = lookup(orig, var, "OR")
        o_lo   = lookup(orig, var, "CI_low")
        o_hi   = lookup(orig, var, "CI_high")
        o_p    = lookup(orig, var, "p_value")
        o_sig  = bool(not pd.isna(o_p) and o_p < 0.05)
        o_ci   = f"{o_lo:.3f}–{o_hi:.3f}" if not pd.isna(o_lo) else "—"

        # Updated model (10-predictor, waist + triglycerides)
        u_or   = lookup(updated, var, "OR")
        u_lo   = lookup(updated, var, "CI_low")
        u_hi   = lookup(updated, var, "CI_high")
        u_p    = lookup(updated, var, "p_value")
        u_sig  = bool(not pd.isna(u_p) and u_p < 0.05)
        u_ci   = f"{u_lo:.3f}–{u_hi:.3f}" if not pd.isna(u_lo) else "—"

        # Extended hypothesis test
        e_or   = lookup(ext, var, "OR")
        e_lo   = lookup(ext, var, "CI_low")
        e_hi   = lookup(ext, var, "CI_high")
        e_p    = lookup(ext, var, "p_value")
        e_sig  = bool(not pd.isna(e_p) and e_p < 0.05)
        e_ci   = f"{e_lo:.3f}–{e_hi:.3f}" if not pd.isna(e_lo) else "—"

        rows.append({
            "variable":               var,
            "label":                  LABEL.get(var, var),
            "category":               CATEGORY.get(var, "Other"),
            # ── Original 9-predictor model ──────────────────────────────────
            "orig_OR":                round(o_or, 3) if not pd.isna(o_or) else "—",
            "orig_95CI":              o_ci,
            "orig_p":                 round(o_p, 4)  if not pd.isna(o_p)  else "—",
            "orig_sig":               o_sig if not pd.isna(o_or) else "—",
            "orig_stars":             sig_star(o_p)  if not pd.isna(o_p)  else "—",
            # ── Updated 10-predictor model (waist + triglycerides) ──────────
            "updated_OR":             round(u_or, 3) if not pd.isna(u_or) else "—",
            "updated_95CI":           u_ci,
            "updated_p":              round(u_p, 4)  if not pd.isna(u_p)  else "—",
            "updated_sig":            u_sig if not pd.isna(u_or) else "—",
            "updated_stars":          sig_star(u_p)  if not pd.isna(u_p)  else "—",
            # ── Extended hypothesis test ─────────────────────────────────────
            "extended_OR":            round(e_or, 3) if not pd.isna(e_or) else "—",
            "extended_95CI":          e_ci,
            "extended_p":             round(e_p, 4)  if not pd.isna(e_p)  else "—",
            "extended_sig":           e_sig if not pd.isna(e_or) else "—",
            "extended_stars":         sig_star(e_p)  if not pd.isna(e_p)  else "—",
            # ── Overall significance flag ────────────────────────────────────
            "significant_in_any_model": (
                o_sig if isinstance(o_sig, bool) else False
            ) or (
                u_sig if isinstance(u_sig, bool) else False
            ) or (
                e_sig if isinstance(e_sig, bool) else False
            ),
        })

    out = pd.DataFrame(rows)

    # Sort: category order, then by significance
    CAT_ORDER = ["Demographics", "Anthropometrics", "Clinical", "Behavioral", "Social", "Environmental"]
    out["_cat_rank"] = out["category"].apply(lambda c: CAT_ORDER.index(c) if c in CAT_ORDER else 99)
    out = out.sort_values(["_cat_rank", "variable"]).drop(columns="_cat_rank").reset_index(drop=True)

    out_path = REPORTS_DIR / "hypothesis_test_all_variables.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("COMPREHENSIVE HYPOTHESIS TEST SUMMARY — NHANES 2017-2018 Liver Injury Analysis")
    print("═" * 90)
    print(f"{'Variable':<28} {'Category':<16} {'Orig OR':>8} {'p':>7} {'Upd OR':>8} {'p':>7} {'Ext OR':>8} {'p':>7} {'Sig?'}")
    print("-" * 90)
    for _, row in out.iterrows():
        o = f"{row['orig_OR']:>8}" if row['orig_OR'] != '—' else f"{'—':>8}"
        op = f"{row['orig_p']:>7}" if row['orig_p'] != '—' else f"{'—':>7}"
        u = f"{row['updated_OR']:>8}" if row['updated_OR'] != '—' else f"{'—':>8}"
        up = f"{row['updated_p']:>7}" if row['updated_p'] != '—' else f"{'—':>7}"
        e = f"{row['extended_OR']:>8}" if row['extended_OR'] != '—' else f"{'—':>8}"
        ep = f"{row['extended_p']:>7}" if row['extended_p'] != '—' else f"{'—':>7}"
        sig = "✓ YES" if row['significant_in_any_model'] else "no"
        print(f"{row['label']:<28} {row['category']:<16} {o} {op} {u} {up} {e} {ep}  {sig}")
    print("─" * 90)
    print("Stars: * p<0.05  ** p<0.01  *** p<0.001   '—' = not included in that model")
    print("Orig = 9-predictor model (BMI); Upd = 10-predictor model (waist+trig); Ext = extended hypothesis test")


if __name__ == "__main__":
    main()
