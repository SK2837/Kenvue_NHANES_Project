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


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = engineer_features(df)

    avail = [f for f in ALL_FEATURES if f in df.columns]
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


def fig_roc_calibration(fpr, tpr, roc_auc: float,
                        y_test, y_prob, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    ax1.plot(fpr, tpr, color="#4C72B0", lw=2,
             label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8)
    ax2.plot(prob_pred, prob_true, "o-", color="#C44E52", lw=2, label="XGBoost")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Calibration Plot (Reliability Diagram)")
    ax2.legend()

    plt.suptitle("Model Performance — XGBoost, ALT > 40 U/L", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── SHAP explanations ─────────────────────────────────────────────────────────

def run_shap(model: XGBClassifier, X_test: pd.DataFrame,
             y_prob: np.ndarray) -> pd.DataFrame:
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

    # Global summary — beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_labeled, max_display=20, show=False)
    plt.title("SHAP Summary — Global Feature Impact on ALT Risk", fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "fig_shap_summary.png"
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
    p = FIGURES_DIR / "fig_shap_importance.png"
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
    p = FIGURES_DIR / "fig_shap_waterfall_high.png"
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
    p = FIGURES_DIR / "fig_shap_waterfall_low.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)

    return importance_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet = PROCESSED_DIR / "analytic_table.parquet"
    if not parquet.exists():
        raise SystemExit("Analytic table not found — run 02_build_dataset.py first")

    df = pd.read_parquet(parquet)
    logger.info("Loaded analytic table: %s rows × %s cols", f"{len(df):,}", df.shape[1])

    # ── Build feature matrix ──────────────────────────────────────────────────
    X, y = build_feature_matrix(df)

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %s rows  |  Test: %s rows", f"{len(X_train):,}", f"{len(X_test):,}")
    logger.info("Train events: %s (%.1f%%)  |  Test events: %s (%.1f%%)",
                int(y_train.sum()), y_train.mean() * 100,
                int(y_test.sum()),  y_test.mean() * 100)

    # ── Feature selection ─────────────────────────────────────────────────────
    selected_features = run_rfecv(X_train, y_train)
    (REPORTS_DIR / "selected_features.txt").write_text("\n".join(selected_features))
    logger.info("Selected features saved → reports/selected_features.txt")

    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    best_model, best_params = tune_xgboost(X_train_sel, y_train)
    logger.info("Best params: %s", best_params)

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics, y_prob, fpr, tpr = evaluate(best_model, X_test_sel, y_test)

    if metrics["roc_auc"] < 0.70:
        logger.warning("AUC %.4f < 0.70 — investigate features or data quality",
                       metrics["roc_auc"])

    metrics["best_params"]       = best_params
    metrics["n_features_selected"] = len(selected_features)
    metrics["n_features_total"]   = X.shape[1]

    out_metrics = REPORTS_DIR / "ml_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved %s", out_metrics)

    fig_roc_calibration(fpr, tpr, metrics["roc_auc"], y_test, y_prob,
                        FIGURES_DIR / "fig_roc_calibration.png")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    importance_df = run_shap(best_model, X_test_sel, y_prob)

    out_shap = REPORTS_DIR / "shap_rankings.csv"
    importance_df.to_csv(out_shap, index=False)
    logger.info("Saved %s", out_shap)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n── ML Results Summary ──────────────────────────────")
    logger.info("AUC-ROC:          %.4f", metrics["roc_auc"])
    logger.info("PR-AUC:           %.4f", metrics["pr_auc"])
    logger.info("Sensitivity:      %.3f", metrics["sensitivity"])
    logger.info("Specificity:      %.3f", metrics["specificity"])
    logger.info("Optimal threshold:%.4f", metrics["optimal_threshold"])
    logger.info("Top 5 features by SHAP:")
    for _, row in importance_df.head(5).iterrows():
        logger.info("  %-25s  %.4f", row["label"], row["mean_abs_shap"])
    logger.info("ML pipeline complete.")


if __name__ == "__main__":
    main()
