"""
ML Model Comparison: 2017-2018 vs 2021-2023
============================================
Trains the same XGBoost classifier (fixed hyperparameters from 2017-2018
grid search) on the 2021-2023 cohort and compares:
  - Model performance (AUC-ROC, PR-AUC)
  - SHAP feature importance rankings
  - Top predictor shifts between cohorts

Uses same feature set as 05_predict_ml.py for fair comparison.

Outputs:
  reports/ml_metrics_2021.json
  reports/shap_rankings_2021.csv
  reports/shap_comparison.csv          — ranked comparison both cohorts
  figures/fig_shap_comparison.png      — side-by-side bar chart
  figures/fig_roc_comparison.png       — ROC curves overlaid
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR   = Path(__file__).parent.parent / "reports" / "phase3_covid"
FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "phase3_covid"

# Fixed hyperparameters from 2017-2018 grid search (ml_metrics.json)
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 100,
    "subsample": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

# Feature set (same as 05_predict_ml.py)
DERIVED_FEATURES = [
    "RIDAGEYR", "sex_male", "INDFMPIR", "BMXBMI", "BMXWAIST",
    "diabetes", "ever_smoker", "ALQ121",
]
BIOPRO_FEATURES = [
    "LBXSAL", "LBXSAPSI", "LBXSASSI", "LBXSC3SI", "LBXSBU",
    "LBXSCLSI", "LBXSCK", "LBXSCR", "LBXSGB", "LBXSGL",
    "LBXSGTSI", "LBXSIR", "LBXSOSSI", "LBXSPH", "LBXSTB",
    "LBXSCA", "LBXSCH", "LBXSTP", "LBXSTR", "LBXSUA",
]
METAL_FEATURES = ["log_lead", "log_cadmium", "log_mercury"]
ALL_FEATURES   = DERIVED_FEATURES + BIOPRO_FEATURES + METAL_FEATURES

LOG_TRANSFORM = {
    "LBXSBU", "LBXSCK", "LBXSGL", "LBXSGTSI", "LBXSIR",
    "LBXSTB", "LBXSTR", "LBXSUA", "LBXSAPSI", "LBXSASSI",
}

FEATURE_LABELS = {
    "RIDAGEYR": "Age", "sex_male": "Male sex", "INDFMPIR": "Poverty-income ratio",
    "BMXBMI": "BMI", "BMXWAIST": "Waist circumference", "diabetes": "Diabetes",
    "ever_smoker": "Ever smoker", "ALQ121": "Drinking frequency",
    "LBXSAL": "Albumin", "LBXSAPSI": "ALP", "LBXSASSI": "AST",
    "LBXSC3SI": "Bicarbonate", "LBXSBU": "BUN", "LBXSCLSI": "Chloride",
    "LBXSCK": "Creatine kinase", "LBXSCR": "Creatinine", "LBXSGB": "Globulin",
    "LBXSGL": "Glucose", "LBXSGTSI": "GGT", "LBXSIR": "Iron",
    "LBXSOSSI": "Osmolality", "LBXSPH": "Phosphorus", "LBXSTB": "Total bilirubin",
    "LBXSCA": "Calcium", "LBXSCH": "Cholesterol", "LBXSTP": "Total protein",
    "LBXSTR": "Triglycerides", "LBXSUA": "Uric acid",
    "log_lead": "log(Blood Lead)", "log_cadmium": "log(Blood Cadmium)",
    "log_mercury": "log(Blood Mercury)",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex_male"]    = (out["RIAGENDR"] == 1).astype(float)
    out["diabetes"]    = (out["DIQ010"] == 1).astype(float)
    out.loc[out["DIQ010"].isna(), "diabetes"] = np.nan
    out["ever_smoker"] = (out["SMQ020"] == 1).astype(float)
    out.loc[out["SMQ020"].isna(), "ever_smoker"] = np.nan
    out["log_lead"]    = np.log(out["LBXBPB"].clip(lower=0.001))
    out["log_cadmium"] = np.log(out["LBXBCD"].clip(lower=0.001))
    out["log_mercury"] = np.log(out["LBXTHG"].clip(lower=0.001))
    for col in LOG_TRANSFORM:
        if col in out.columns:
            out[f"log_{col}"] = np.log(out[col].clip(lower=0.001))
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = engineer_features(df)
    avail = [f for f in ALL_FEATURES if f in df.columns]
    X = df[avail].copy()
    y = df["ALT_elevated_40"].copy()
    valid = y.notna()
    X, y = X[valid], y[valid]
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_arr, columns=avail, index=X.index)
    return X, y.reset_index(drop=True)


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, cohort: str,
) -> tuple[XGBClassifier, pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("[%s] Train: %d  Test: %d  Events in test: %d",
                cohort, len(X_train), len(X_test), int(y_test.sum()))

    model = XGBClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_a  = roc_auc_score(y_test, y_prob)
    pr_a   = average_precision_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    logger.info("[%s] AUC-ROC=%.4f  PR-AUC=%.4f", cohort, roc_a, pr_a)

    metrics = {
        "cohort": cohort,
        "roc_auc": round(roc_a, 4),
        "pr_auc": round(pr_a, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_events_test": int(y_test.sum()),
        "n_features": X.shape[1],
    }

    # SHAP values (TreeExplainer — fast)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    shap_df    = pd.DataFrame({
        "feature": X.columns,
        "label":   [FEATURE_LABELS.get(f, f) for f in X.columns],
        "mean_abs_shap": mean_shap,
        "cohort": cohort,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1

    return model, shap_df, metrics, fpr, tpr, y_prob


def make_shap_comparison_plot(shap17: pd.DataFrame, shap21: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    # Use union of top-N from each cohort
    top17 = set(shap17.head(top_n)["feature"].tolist())
    top21 = set(shap21.head(top_n)["feature"].tolist())
    features = list(top17 | top21)

    d17 = shap17.set_index("feature")["mean_abs_shap"].to_dict()
    d21 = shap21.set_index("feature")["mean_abs_shap"].to_dict()
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    # Sort by average importance
    avg = [(d17.get(f, 0) + d21.get(f, 0)) / 2 for f in features]
    order = sorted(range(len(features)), key=lambda i: avg[i])
    features = [features[i] for i in order]
    labels   = [labels[i] for i in order]
    vals_17  = [d17.get(f, 0) for f in features]
    vals_21  = [d21.get(f, 0) for f in features]

    y = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.55)))
    ax.barh(y + width/2, vals_17, width, color="#4C72B0", label="2017-2018 (pre-COVID)", alpha=0.85)
    ax.barh(y - width/2, vals_21, width, color="#DD8452", label="2021-2023 (post-COVID)", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        f"XGBoost SHAP Feature Importance — Pre vs Post COVID (Top {top_n})\n"
        "(same model hyperparameters applied to each cohort independently)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def make_roc_comparison_plot(
    fpr17: np.ndarray, tpr17: np.ndarray, auc17: float,
    fpr21: np.ndarray, tpr21: np.ndarray, auc21: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr17, tpr17, color="#4C72B0", lw=2,
            label=f"2017-2018 pre-COVID (AUC={auc17:.3f})")
    ax.plot(fpr21, tpr21, color="#DD8452", lw=2, linestyle="--",
            label=f"2021-2023 post-COVID (AUC={auc21:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: XGBoost Pre vs Post COVID", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
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

    logger.info("Building 2017-2018 feature matrix...")
    df17 = pd.read_parquet(parquet_17)
    X17, y17 = build_feature_matrix(df17)
    logger.info("2017-2018: %d rows × %d features  |  events: %d (%.1f%%)",
                len(X17), X17.shape[1], int(y17.sum()), y17.mean() * 100)

    logger.info("Building 2021-2023 feature matrix...")
    df21 = pd.read_parquet(parquet_21)
    X21, y21 = build_feature_matrix(df21)
    logger.info("2021-2023: %d rows × %d features  |  events: %d (%.1f%%)",
                len(X21), X21.shape[1], int(y21.sum()), y21.mean() * 100)

    # Align feature columns (use intersection)
    common_features = [f for f in X17.columns if f in X21.columns]
    X17 = X17[common_features]
    X21 = X21[common_features]
    logger.info("Common features: %d", len(common_features))

    logger.info("\n── Training on 2017-2018 ─────────────────────────────────")
    model17, shap17, metrics17, fpr17, tpr17, prob17 = train_and_evaluate(X17, y17, "2017-2018")

    logger.info("\n── Training on 2021-2023 ─────────────────────────────────")
    model21, shap21, metrics21, fpr21, tpr21, prob21 = train_and_evaluate(X21, y21, "2021-2023")

    # Save individual outputs
    shap17.to_csv(REPORTS_DIR / "shap_rankings_comparison_2017.csv", index=False)
    shap21.to_csv(REPORTS_DIR / "shap_rankings_2021.csv", index=False)

    with open(REPORTS_DIR / "ml_metrics_2021.json", "w") as f:
        json.dump(metrics21, f, indent=2)

    # Combined SHAP comparison
    shap_comp = shap17[["feature", "label", "mean_abs_shap", "rank"]].rename(
        columns={"mean_abs_shap": "shap_2017", "rank": "rank_2017"}
    ).merge(
        shap21[["feature", "mean_abs_shap", "rank"]].rename(
            columns={"mean_abs_shap": "shap_2021", "rank": "rank_2021"}
        ),
        on="feature", how="outer",
    )
    shap_comp["rank_change"] = shap_comp["rank_2017"] - shap_comp["rank_2021"]
    shap_comp = shap_comp.sort_values("shap_2017", ascending=False).reset_index(drop=True)
    shap_comp.to_csv(REPORTS_DIR / "shap_comparison.csv", index=False)
    logger.info("Saved reports/shap_comparison.csv")

    # Plots
    make_shap_comparison_plot(shap17, shap21, FIGURES_DIR / "fig_shap_comparison.png")
    make_roc_comparison_plot(
        fpr17, tpr17, metrics17["roc_auc"],
        fpr21, tpr21, metrics21["roc_auc"],
        FIGURES_DIR / "fig_roc_comparison.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n══════ ML COMPARISON SUMMARY ══════\n")
    logger.info("  %-25s  %-10s  %-10s", "Metric", "2017-2018", "2021-2023")
    logger.info("  " + "-" * 50)
    logger.info("  %-25s  %-10.4f  %-10.4f", "AUC-ROC",    metrics17["roc_auc"], metrics21["roc_auc"])
    logger.info("  %-25s  %-10.4f  %-10.4f", "PR-AUC",     metrics17["pr_auc"],  metrics21["pr_auc"])
    logger.info("  %-25s  %-10d  %-10d",    "N features",  metrics17["n_features"], metrics21["n_features"])
    logger.info("  %-25s  %-10d  %-10d",    "N train",     metrics17["n_train"],  metrics21["n_train"])
    logger.info("  %-25s  %-10d  %-10d",    "N test",      metrics17["n_test"],   metrics21["n_test"])

    logger.info("\n── Top 10 SHAP Features ──────────────────────────────────")
    logger.info("  %-5s  %-25s  %-10s  %-10s  %-10s", "Rank", "Feature", "SHAP-17", "SHAP-21", "Δ rank")
    logger.info("  " + "-" * 60)
    for _, row in shap_comp.head(10).iterrows():
        logger.info(
            "  %-5.0f  %-25s  %-10.4f  %-10.4f  %+.0f",
            row.get("rank_2017", np.nan),
            str(row["label"])[:25],
            row.get("shap_2017", 0),
            row.get("shap_2021", 0),
            row.get("rank_change", 0),
        )

    logger.info("\nNext step: run python src/17_comparison_figures.py")


if __name__ == "__main__":
    main()
