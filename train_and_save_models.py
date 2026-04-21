"""
Train best fraud and severity models from insurance_claims_fraud.csv and save them separately into a folder.

Flow:
  1. Data preprocessing (load, clean, prepare features)
  2. Train/Test split (stratified)
  3. Train on natural (imbalanced) data; class weights for fraud
  4. Train multiple models (fraud: RF, XGB, LGB, GB; severity: RF, XGB, LGB)
  5. Compare metrics (CV F1 for fraud, CV R² for severity)
  6. Select best model for each task
  7. Save best models into trained_models/ folder

Usage:
  python train_and_save_models.py

Output folder: trained_models/
  - best_fraud_<ModelName>.pkl   (best fraud classifier)
  - best_severity_<ModelName>.pkl (best severity regressor)
  - full_store.pkl              (complete store for the app: encoders, feature_names, etc.)
  - metrics_summary.txt          (comparison of all models)
"""

import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from typing import cast

# Use existing pipeline for preprocessing and model definitions
from ml_pipeline import (
    load_and_clean_data,
    prepare_features,
    run_fraud_pipeline,
    run_severity_pipeline,
    _ensure_no_nan,
    SEED,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models")


def get_data_path():
    """Only insurance_claims_fraud.csv in uploads."""
    return os.path.join(UPLOADS_DIR, "insurance_claims_fraud.csv")


def main():
    print("=" * 60)
    print("Train best models - flow: Preprocess -> Split -> Imbalance -> Train -> Compare -> Save")
    print("=" * 60)

    csv_path = get_data_path()
    if not os.path.exists(csv_path):
        print("Missing insurance_claims_fraud.csv. Place it in uploads/.")
        sys.exit(1)
    print(f"\n1. Data: {csv_path}")

    # --- Data preprocessing ---
    print("\n2. Data preprocessing (load, clean, prepare features)...")
    df = load_and_clean_data(csv_path)
    if df is None or len(df) < 200:
        print("Not enough data (need at least 200 rows).")
        sys.exit(1)
    prep = prepare_features(df)
    if prep is None:
        print("Feature preparation failed (e.g. fraud column has only one class).")
        sys.exit(1)
    X, y_fraud, y_amount, encoders, feature_names, cat_cols, cat_modes = prep
    # Use SimpleImputer without keep_empty_features to avoid version-specific internals
    imputer = SimpleImputer(strategy="median", fill_value=0)
    X_imputed = imputer.fit_transform(X)
    X_imputed = _ensure_no_nan(X_imputed)
    print(f"   Features: {len(feature_names)}, Samples: {len(X_imputed)}")

    # --- Train/Test split ---
    print("\n3. Train/Test split (stratified, test_size=0.2)...")
    X_train, X_test, yf_train, yf_test, ya_train, ya_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        train_test_split(
            X_imputed, y_fraud, y_amount, test_size=TEST_SIZE, random_state=SEED, stratify=y_fraud
        ),
    )
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    yf_train = pd.Series(yf_train)
    yf_test = pd.Series(yf_test)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    # Log-transform claim amount for severity, same as ml_pipeline.run_full_pipeline
    ya_train_arr = np.asarray(ya_train, dtype=float)
    ya_test_arr = np.asarray(ya_test, dtype=float)
    ya_train_log = np.log1p(ya_train_arr)
    ya_test_log = np.log1p(ya_test_arr)

    # --- Handle imbalance + Train multiple models + Compare + Select best ---
    print("\n4. Fraud: train on natural class distribution (class weights) -> compare (CV F1) -> select best")
    best_fraud, best_fraud_name, fraud_metrics = run_fraud_pipeline(X_train, X_test, yf_train, yf_test)
    print(f"   Best fraud model: {best_fraud_name}")

    print("\n5. Severity: train models -> compare (CV R2) -> select best")
    best_severity, best_severity_name, severity_metrics = run_severity_pipeline(X_train, X_test, ya_train_log, ya_test_log)
    print(f"   Best severity model: {best_severity_name}")

    # Fraud threshold (same as in ml_pipeline)
    try:
        if best_fraud is not None:
            probs = best_fraud.predict_proba(_ensure_no_nan(X_test))[:, 1]  # type: ignore[union-attr]
        else:
            probs = np.zeros(len(X_test))
        best_f1, best_rec, fraud_threshold = 0.0, 0.0, 0.35
        for t in np.linspace(0.18, 0.55, 20):
            pred_t = (probs >= t).astype(int)
            f1_t = f1_score(yf_test, pred_t, zero_division=0)  # type: ignore[arg-type]
            rec_t = recall_score(yf_test, pred_t, zero_division=0)  # type: ignore[arg-type]
            if f1_t > best_f1 or (f1_t == best_f1 and rec_t > best_rec):
                best_f1, best_rec, fraud_threshold = f1_t, rec_t, float(t)
    except Exception:
        fraud_threshold = 0.35

    train_means = {c: float(v) if pd.notna(v) else 0.0 for c, v in pd.DataFrame(X_imputed, columns=feature_names).mean().items()}

    # --- Save best models separately into folder ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Safe filename: replace spaces with underscores
    fraud_name = best_fraud_name or "Model"
    sev_name = best_severity_name or "Model"
    fraud_fname = f"best_fraud_{fraud_name.replace(' ', '_')}.pkl"
    sev_fname = f"best_severity_{sev_name.replace(' ', '_')}.pkl"
    fraud_path = os.path.join(OUTPUT_DIR, fraud_fname)
    sev_path = os.path.join(OUTPUT_DIR, sev_fname)

    joblib.dump(best_fraud, fraud_path)
    joblib.dump(best_severity, sev_path)
    print(f"\n6. Saved best models into folder: {OUTPUT_DIR}")
    print(f"   - {fraud_fname}")
    print(f"   - {sev_fname}")

    # Full store (so the app can load one file from folder and use it)
    full_store = {
        "fraud_model": best_fraud,
        "severity_model": best_severity,
        "feature_names": feature_names,
        "train_means": train_means,
        "imputer": imputer,
        "encoders": encoders,
        "cat_cols": cat_cols,
        "cat_modes": cat_modes,
        "fraud_threshold": fraud_threshold,
        "fraud_metrics": fraud_metrics,
        "severity_metrics": severity_metrics,
        "best_fraud_name": best_fraud_name,
        "best_severity_name": best_severity_name,
        # Flag so app knows severity model was trained on log(amount)
        "severity_log_target": True,
    }
    full_store_path = os.path.join(OUTPUT_DIR, "full_store.pkl")
    joblib.dump(full_store, full_store_path)
    print(f"   - full_store.pkl (for app: encoders, feature_names, both models)")

    # Metrics summary file
    lines = [
        "Training summary (insurance_claims_fraud)",
        "=" * 50,
        f"Best fraud model: {best_fraud_name}",
        f"Best severity model: {best_severity_name}",
        "",
        "Fraud models (CV F1 = selection; Test = held-out):",
    ]
    for name, m in fraud_metrics.items():
        cv = m.get("cv_f1_mean", "")
        lines.append(f"  {name}: CV_F1={cv}, Test Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, Recall={m['recall']:.4f}, ROC-AUC={m['roc_auc']:.4f}")
    lines.append("")
    lines.append("Severity models (CV R² = selection; Test = held-out):")
    for name, m in severity_metrics.items():
        cv = m.get("cv_r2_mean", "")
        lines.append(f"  {name}: CV_R2={cv}, Test MAE=${m['mae']:.0f}, R²={m['r2']:.4f}")
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   - metrics_summary.txt")

    print("\nDone. Use the app with models from this folder by pointing it to trained_models/full_store.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
