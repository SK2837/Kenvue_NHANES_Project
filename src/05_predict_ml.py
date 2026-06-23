"""
XGBoost + SHAP prediction model for elevated ALT (NHANES 2017-2018).

Separates prediction from population inference:
  - Inference (04_inference.py): WHY is ALT elevated at population level?
  - Prediction (this script):    WHO is at individual risk?

Outputs:
  reports/ml_metrics.json
  reports/selected_features.txt
  reports/shap_rankings.csv
  figures/fig_shap_summary.png
  figures/fig_shap_importance.png
  figures/fig_shap_waterfall_high.png
  figures/fig_shap_waterfall_low.png
  figures/fig_roc_calibration.png
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


def _binary_roc_auc(y_true, y_score):
    if hasattr(y_score, "ndim") and y_score.ndim == 2:
        y_score = y_score[:, 1]
    return roc_auc_score(y_true, y_score)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase2_analysis"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase2_analysis"

# ── Feature definitions ────────────────────────────────────────────────────────

# Demographic / behavioural (derived)
DERIVED_FEATURES = [
    "RIDAGEYR",    # age
    "sex_male",    # 1=male
    "INDFMPIR",    # poverty-income ratio
    "BMXBMI",      # BMI
    "BMXWAIST",    # waist circumference
    "diabetes",    # 1=diabetes diagnosed
    "ever_smoker", # 1=ever smoked 100+ cigarettes
    "ALQ121",      # drinking frequency
]

# Full biochemistry panel (BIOPRO_J) — exclude ALT itself and duplicate SI-unit cols
# Keep primary measurement columns only (LBX prefix = measured value)
BIOPRO_FEATURES = [
    "LBXSAL",    # albumin
    "LBXSAPSI",  # ALP
    "LBXSASSI",  # AST
    "LBXSC3SI",  # bicarbonate (CO2)
    "LBXSBU",    # BUN (urea)
    "LBXSCLSI",  # chloride
    "LBXSCK",    # creatine kinase
    "LBXSCR",    # creatinine
    "LBXSGB",    # globulin
    "LBXSGL",    # glucose
    "LBXSGTSI",  # GGT
    "LBXSIR",    # iron
    "LBXSOSSI",  # osmolality
    "LBXSPH",    # phosphorus
    "LBXSTB",    # total bilirubin
    "LBXSCA",    # calcium
    "LBXSCH",    # total cholesterol
    "LBXSTP",    # total protein
    "LBXSTR",    # triglycerides
    "LBXSUA",    # uric acid
]

# Heavy metals (log-transformed)
METAL_FEATURES = ["log_lead", "log_cadmium", "log_mercury"]

ALL_FEATURES = DERIVED_FEATURES + BIOPRO_FEATURES + METAL_FEATURES

# Liver co-travelers: other hepatic markers from the SAME blood draw as ALT.
# Including these causes target leakage — the model learns "AST is high → ALT is high"
# rather than learning genuine risk factors. Excluded from the primary clean model;
# retained only in the leaky benchmark for transparency.
LIVER_COTRAVELERS = {
    "LBXSASSI",  # AST       — hepatocellular enzyme, r=0.73 with ALT
    "LBXSGTSI",  # GGT       — hepatobiliary enzyme
    "LBXSAPSI",  # ALP       — hepatobiliary / bone enzyme
    "LBXSAL",    # Albumin   — synthesised by liver; reflects chronic function
    "LBXSGB",    # Globulin  — complement to albumin in total protein
    "LBXSTB",    # Total bilirubin — hepatic conjugation product
    "LBXSTP",    # Total protein  — albumin + globulin combined
}

CLEAN_FEATURES = [f for f in ALL_FEATURES if f not in LIVER_COTRAVELERS]

# Right-skewed biomarkers to log-transform
LOG_TRANSFORM = {
    "LBXSBU", "LBXSCK", "LBXSGL", "LBXSGTSI", "LBXSIR",
    "LBXSTB", "LBXSTR", "LBXSUA", "LBXSAPSI", "LBXSASSI",
}

# Human-readable labels for figures
FEATURE_LABELS = {
    "RIDAGEYR":  "Age",
    "sex_male":  "Male sex",
    "INDFMPIR":  "Poverty-income ratio",
    "BMXBMI":    "BMI",
    "BMXWAIST":  "Waist circumference",
    "diabetes":  "Diabetes",
    "ever_smoker":"Ever smoker",
    "ALQ121":    "Drinking frequency",
    "LBXSAL":    "Albumin",
    "LBXSAPSI":  "ALP",
    "LBXSASSI":  "AST",
    "LBXSC3SI":  "Bicarbonate",
    "LBXSBU":    "BUN",
    "LBXSCLSI":  "Chloride",
    "LBXSCK":    "Creatine kinase",
    "LBXSCR":    "Creatinine",
    "LBXSGB":    "Globulin",
    "LBXSGL":    "Glucose",
    "LBXSGTSI":  "GGT",
    "LBXSIR":    "Iron",
    "LBXSOSSI":  "Osmolality",
    "LBXSPH":    "Phosphorus",
    "LBXSTB":    "Total bilirubin",
    "LBXSCA":    "Calcium",
    "LBXSCH":    "Cholesterol",
    "LBXSTP":    "Total protein",
    "LBXSTR":    "Triglycerides",
    "LBXSUA":    "Uric acid",
    "log_lead":  "log(Blood Lead)",
    "log_cadmium":"log(Blood Cadmium)",
    "log_mercury":"log(Blood Mercury)",
}


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sex_male"]    = (out["RIAGENDR"] == 1).astype(float)
    out["diabetes"]    = (out["DIQ010"]   == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan
    out["ever_smoker"] = (out["SMQ020"]   == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan

    out["log_lead"]    = np.log(out["LBXBPB"])
    out["log_cadmium"] = np.log(out["LBXBCD"])
    out["log_mercury"] = np.log(out["LBXTHG"])

    for col in LOG_TRANSFORM:
        if col in out.columns:
            vals = out[col]
            out[f"log_{col}"] = np.log(vals.clip(lower=0.001))

    return out


def build_feature_matrix(
    df: pd.DataFrame, features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    df = engineer_features(df)

    feature_list = features if features is not None else ALL_FEATURES
    avail = [f for f in feature_list if f in df.columns]
    X = df[avail].copy()
    y = df["ALT_elevated_40"].copy()

    # Drop rows missing the outcome
    valid = y.notna()
    X, y = X[valid], y[valid]

    # Median imputation for remaining missing values
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_arr, columns=avail, index=X.index)

    logger.info("Feature matrix: %s rows × %s cols  |  Events: %s (%.1f%%)",
                f"{len(X):,}", X.shape[1], int(y.sum()), y.mean() * 100)
    return X, y


# ── Feature selection via RFECV ───────────────────────────────────────────────

def run_rfecv(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    logger.info("Running RFECV (5-fold CV, optimising AUC-ROC) ...")
    base_xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    selector = RFECV(
        estimator=base_xgb, step=1, cv=cv,
        scoring=make_scorer(_binary_roc_auc, response_method="predict_proba"),
        min_features_to_select=5, n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    logger.info("RFECV selected %d / %d features", len(selected), X_train.shape[1])
    return selected


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[XGBClassifier, dict]:
    logger.info("Grid search over XGBoost hyperparameters ...")
    param_grid = {
        "n_estimators":  [100, 300],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    xgb = XGBClassifier(
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    logger.info("Best params: %s  |  CV AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, grid.best_params_


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: XGBClassifier, X_test: pd.DataFrame,
             y_test: pd.Series) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)

    # Youden's J optimal threshold
    j_scores = tpr - fpr
    opt_idx  = np.argmax(j_scores)
    opt_thr  = float(thresholds[opt_idx])
    y_pred   = (y_prob >= opt_thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "roc_auc":         round(float(roc_auc), 4),
        "pr_auc":          round(float(pr_auc), 4),
        "optimal_threshold": round(opt_thr, 4),
        "sensitivity":     round(float(tp / (tp + fn)), 4),
        "specificity":     round(float(tn / (tn + fp)), 4),
        "ppv":             round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0,
        "npv":             round(float(tn / (tn + fn)), 4) if (tn + fn) > 0 else 0.0,
        "n_test":          int(len(y_test)),
        "n_events_test":   int(y_test.sum()),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    }
    logger.info("AUC-ROC: %.4f  |  PR-AUC: %.4f  |  Sensitivity: %.3f  |  Specificity: %.3f",
                roc_auc, pr_auc, metrics["sensitivity"], metrics["specificity"])
    return metrics, y_prob, fpr, tpr


def fig_roc_calibration(
    fpr, tpr, roc_auc: float,
    y_test, y_prob, path: Path,
    fpr_bench=None, tpr_bench=None, roc_auc_bench: float | None = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve — clean model
    ax1.plot(fpr, tpr, color="#4C72B0", lw=2,
             label=f"Clean risk-factor model (AUC = {roc_auc:.3f})")
    if fpr_bench is not None:
        ax1.plot(fpr_bench, tpr_bench, color="#C44E52", lw=2, ls="--",
                 label=f"Full biochemistry benchmark* (AUC = {roc_auc_bench:.3f})")
        ax1.text(0.98, 0.08,
                 "*benchmark includes liver co-travelers (AST, GGT, ALP…)\n"
                 " — not a valid risk-factor model",
                 transform=ax1.transAxes, fontsize=7, color="#C44E52",
                 ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", fontsize=8)

    # Calibration — clean model only
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8)
    ax2.plot(prob_pred, prob_true, "o-", color="#4C72B0", lw=2,
             label="Clean model")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Calibration Plot — Clean Model")
    ax2.legend()

    plt.suptitle("Model Performance — XGBoost, ALT > 40 U/L\n"
                 "(Clean model excludes liver co-travelers to avoid target leakage)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── SHAP explanations ─────────────────────────────────────────────────────────

def run_shap(model: XGBClassifier, X_test: pd.DataFrame,
             y_prob: np.ndarray, fig_suffix: str = "") -> pd.DataFrame:
    logger.info("Computing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Use human-readable column names for plots
    X_labeled = X_test.rename(columns=FEATURE_LABELS)
    shap_labeled = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=X_labeled.values,
        feature_names=X_labeled.columns.tolist(),
    )

    sfx = fig_suffix  # e.g. "" for primary, "_benchmark" for leaky model

    # Global summary — beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_labeled, max_display=20, show=False)
    plt.title("SHAP Summary — Global Feature Impact on ALT Risk", fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / f"fig_shap_summary{sfx}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)

    # Feature importance bar chart (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "label":   [FEATURE_LABELS.get(c, c) for c in X_test.columns],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    top20 = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top20["label"][::-1], top20["mean_abs_shap"][::-1],
                   color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features by SHAP Importance", fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / f"fig_shap_importance{sfx}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)

    # Waterfall — highest-risk individual in test set
    high_idx = int(np.argmax(y_prob))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_labeled[high_idx], max_display=12, show=False)
    plt.title(f"High-Risk Individual — Predicted P(ALT>40) = {y_prob[high_idx]:.1%}",
              fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / f"fig_shap_waterfall_high{sfx}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)

    # Waterfall — lowest-risk individual
    low_idx = int(np.argmin(y_prob))
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_labeled[low_idx], max_display=12, show=False)
    plt.title(f"Low-Risk Individual — Predicted P(ALT>40) = {y_prob[low_idx]:.1%}",
              fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / f"fig_shap_waterfall_low{sfx}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)

    return importance_df


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_pipeline(
    X: pd.DataFrame, y: pd.Series, label: str, fig_suffix: str = ""
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Train, tune, evaluate one model. Returns (metrics, y_prob, fpr, tpr, shap_df)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("[%s] Train: %s  |  Test: %s", label, f"{len(X_train):,}", f"{len(X_test):,}")

    selected = run_rfecv(X_train, y_train)
    X_train_sel = X_train[selected]
    X_test_sel  = X_test[selected]

    best_model, best_params = tune_xgboost(X_train_sel, y_train)
    metrics, y_prob, fpr, tpr = evaluate(best_model, X_test_sel, y_test)
    metrics["best_params"]        = best_params
    metrics["n_features_selected"]= len(selected)
    metrics["n_features_total"]   = X.shape[1]
    metrics["model_label"]        = label

    importance_df = run_shap(best_model, X_test_sel, y_prob, fig_suffix=fig_suffix)
    return metrics, y_prob, fpr, tpr, importance_df, y_test, selected


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        raise SystemExit("Analytic table not found — run 02_build_dataset.py first")

    df = pd.read_parquet(parquet)
    logger.info("Loaded analytic table: %s rows × %s cols", f"{len(df):,}", df.shape[1])

    # ── Model 1: CLEAN — no liver co-travelers (primary, honest model) ────────
    logger.info("\n═══ PRIMARY: clean risk-factor model (liver co-travelers excluded) ═══")
    X_clean, y = build_feature_matrix(df, features=CLEAN_FEATURES)
    (
        metrics_clean, y_prob_clean, fpr_clean, tpr_clean,
        shap_clean, y_test_clean, selected_clean,
    ) = _run_pipeline(X_clean, y, label="clean", fig_suffix="")

    (REPORTS_DIR / "selected_features.txt").write_text("\n".join(selected_clean))
    (REPORTS_DIR / "ml_metrics.json").write_text(json.dumps(metrics_clean, indent=2))
    shap_clean.to_csv(REPORTS_DIR / "shap_rankings.csv", index=False)
    logger.info("[clean] AUC-ROC: %.4f", metrics_clean["roc_auc"])

    # ── Model 2: BENCHMARK — full biochemistry (leaky, for transparency only) ─
    logger.info("\n═══ BENCHMARK: full biochemistry model (includes liver co-travelers) ═══")
    logger.info("NOTE: this model is leaky — AST/GGT predict ALT trivially. For reference only.")
    X_full, y_full = build_feature_matrix(df, features=ALL_FEATURES)
    (
        metrics_bench, y_prob_bench, fpr_bench, tpr_bench,
        shap_bench, y_test_bench, selected_bench,
    ) = _run_pipeline(X_full, y_full, label="benchmark (leaky)", fig_suffix="_benchmark")

    metrics_bench["leakage_warning"] = (
        "Includes liver co-travelers (AST, GGT, ALP, albumin, globulin, bilirubin, total protein). "
        "High AUC reflects biochemical collinearity, not genuine predictive signal. "
        "Use clean model (ml_metrics.json) for all clinical and public-health claims."
    )
    (REPORTS_DIR / "ml_metrics_benchmark.json").write_text(json.dumps(metrics_bench, indent=2))
    shap_bench.to_csv(REPORTS_DIR / "shap_rankings_benchmark.csv", index=False)
    logger.info("[benchmark] AUC-ROC: %.4f  (leaky — do not present as risk-factor model)",
                metrics_bench["roc_auc"])

    # ── ROC figure: clean model + benchmark overlay ───────────────────────────
    fig_roc_calibration(
        fpr_clean, tpr_clean, metrics_clean["roc_auc"],
        y_test_clean, y_prob_clean,
        FIGURES_DIR / "fig_roc_calibration.png",
        fpr_bench=fpr_bench, tpr_bench=tpr_bench,
        roc_auc_bench=metrics_bench["roc_auc"],
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n── ML Results Summary ──────────────────────────────")
    logger.info("CLEAN MODEL (primary — no leakage):")
    logger.info("  AUC-ROC:     %.4f", metrics_clean["roc_auc"])
    logger.info("  PR-AUC:      %.4f", metrics_clean["pr_auc"])
    logger.info("  Sensitivity: %.3f", metrics_clean["sensitivity"])
    logger.info("  Specificity: %.3f", metrics_clean["specificity"])
    logger.info("  Features:    %d selected from %d", metrics_clean["n_features_selected"],
                metrics_clean["n_features_total"])
    logger.info("  Top 5 SHAP predictors:")
    for _, row in shap_clean.head(5).iterrows():
        logger.info("    %-25s  %.4f", row["label"], row["mean_abs_shap"])
    logger.info("")
    logger.info("BENCHMARK (leaky — full biochemistry, for reference only):")
    logger.info("  AUC-ROC:     %.4f  ← inflated by AST/GGT co-travelers",
                metrics_bench["roc_auc"])
    logger.info("ML pipeline complete.")


if __name__ == "__main__":
    main()
