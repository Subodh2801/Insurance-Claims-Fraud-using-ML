"""ML pipeline: fraud detection (classification) and loss severity (regression). Dataset: insurance_claims_fraud.csv."""

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Any, cast
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import clone, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb  # type: ignore[reportMissingImports]
import lightgbm as lgb
import joblib

FRAUD_TARGET = "fraud_reported"
AMOUNT_TARGET = "total_claim_amount"
EXCLUDE = [
    FRAUD_TARGET,
    AMOUNT_TARGET,
    "policy_number",
    "policy_id",  # identifier, not predictive
    "policy_bind_date",
    "incident_date",
    "incident_location",
]


def _is_excluded(col):
    """Exclude identifiers and claim amount columns (data leakage when predicting)."""
    if col in EXCLUDE:
        return True
    c = str(col).lower()
    if "total_claim_amount" in c or "claim_amount" in c:
        return True
    return False


def _is_na_scalar(v: Any) -> bool:
    """Return True if v is None or a scalar NA (e.g. np.nan). Safe for conditionals."""
    if v is None:
        return True
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


SEED = 42
TEST_SIZE = 0.2
N_CV = 5  # Folds for cross-validation (model selection)
# Retrain best model on full data for production (uses all samples = most accurate)
RETRAIN_BEST_ON_FULL_DATA = True


def _fraud_models():
    """Fraud classifiers with stronger regularization to reduce overfitting on easy synthetic data."""
    return [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                min_child_weight=4,
                reg_alpha=0.3,
                reg_lambda=2.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                min_child_samples=20,
                reg_alpha=0.3,
                reg_lambda=2.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                verbose=-1,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=120,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=SEED,
            ),
        ),
    ]


def _severity_models():
    """Return (name, regressor) pairs with stronger regularization to avoid unrealistically high R²."""
    return [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                max_features=cast(Any, "sqrt"),
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                min_child_weight=5,
                reg_alpha=0.3,
                reg_lambda=2.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "LightGBM",
            lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                min_child_samples=20,
                reg_alpha=0.3,
                reg_lambda=2.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                verbose=-1,
            ),
        ),
    ]


def load_and_clean_data(csv_path):
    """Load CSV or Excel, replace ? with NaN, fill missing (mode for categorical, median for numeric)."""
    path = str(csv_path).lower()
    if path.endswith(('.xlsx', '.xls')):
        df = cast(pd.DataFrame, pd.read_excel(csv_path)).replace("?", np.nan)
    else:
        df = cast(pd.DataFrame, pd.read_csv(csv_path)).replace("?", np.nan)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        m = df[col].mode()
        fill_val = m.iloc[0] if len(m) > 0 else "Unknown"
        fill_val = str(fill_val) if pd.notna(fill_val) else "Unknown"
        df[col] = df[col].astype(object).fillna(fill_val)
    for col in df.select_dtypes(include=[np.number]).columns:
        med = df[col].median()
        df[col] = df[col].fillna(med if not _is_na_scalar(med) else 0)
    return df


def prepare_features(df):
    """Build X, y_fraud, y_amount; label-encode categoricals; return encoders and feature names."""
    df = df.copy()
    # Allow alternative fraud/amount column names when working with other datasets (e.g. fraud_oracle.csv)
    # Map known fraud label into FRAUD_TARGET
    if FRAUD_TARGET not in df.columns:
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl in ("fraudfound_p", "fraudfound", "fraud", "fraud_flag", "fraud_indicator"):
                df[FRAUD_TARGET] = df[c]
                break
    # If no amount column, synthesize one from vehicle price bands or fall back to zeros
    if AMOUNT_TARGET not in df.columns:
        if "VehiclePrice" in df.columns or "vehicleprice" in [str(c).lower() for c in df.columns]:
            vp_col = None
            for c in df.columns:
                if str(c).lower() == "vehicleprice":
                    vp_col = c
                    break
            # Rough midpoints for the oracle price bands
            price_map = {
                "less than 20000": 15000,
                "20000 to 29000": 24500,
                "30000 to 39000": 34500,
                "40000 to 49000": 44500,
                "50000 to 59000": 54500,
                "60000 to 69000": 64500,
                "more than 69000": 75000,
            }
            vp_series = df[vp_col].astype(str).str.strip()
            approx_amount = vp_series.map(price_map)
            # If some bands are unknown, fill with overall median of known midpoints
            median_val = approx_amount[approx_amount.notna()].median() if approx_amount.notna().any() else 30000.0
            df[AMOUNT_TARGET] = approx_amount.fillna(median_val).astype(float)
        else:
            df[AMOUNT_TARGET] = 0.0
    fraud_map = {"Y": 1, "N": 0, "YES": 1, "NO": 0, "1": 1, "0": 0, "TRUE": 1, "FALSE": 0}
    df[FRAUD_TARGET] = df[FRAUD_TARGET].astype(str).str.upper().str.strip().map(fraud_map).fillna(0).astype(int)
    if df[FRAUD_TARGET].nunique() < 2:
        return None
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [FRAUD_TARGET, AMOUNT_TARGET] and not _is_excluded(c)]
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c not in EXCLUDE and not _is_excluded(c)]
    use_cols = [c for c in num_cols + cat_cols if c in df.columns]
    X = df[use_cols].copy()
    # Domain feature: impossible / inconsistent combinations (e.g. parked car + total loss or rollover) often indicate fraud
    incident_type_str = df["incident_type"].astype(str).str.lower() if "incident_type" in df.columns else pd.Series("", index=df.index)
    incident_sev_str = df["incident_severity"].astype(str).str.lower() if "incident_severity" in df.columns else pd.Series("", index=df.index)
    collision_str = df["collision_type"].astype(str).str.lower() if "collision_type" in df.columns else pd.Series("", index=df.index)
    parked = incident_type_str.str.contains("parked", na=False)
    total_loss = incident_sev_str.str.contains("total", na=False)
    rollover = collision_str.str.contains("rollover", na=False)
    X["inconsistent_claim"] = ((parked & total_loss) | (parked & rollover)).astype(np.int64)
    # More fraud-signal features: severity = total loss, multi-vehicle (often higher risk)
    X["severity_total_loss"] = total_loss.astype(np.int64)
    if "number_of_vehicles_involved" in X.columns:
        X["multi_vehicle"] = (X["number_of_vehicles_involved"].fillna(0) >= 2).astype(np.int64)
    else:
        X["multi_vehicle"] = 0
    encoders = {}
    for c in cat_cols:
        if c not in X.columns:
            continue
        # Convert to object first so we can assign encoded ints (StringDtype rejects int)
        X[c] = X[c].astype(object)
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    X = X.fillna(X.median())
    X = X.fillna(0)  # Fallback for columns where median was NaN (e.g. constant columns)
    cat_modes = {}
    for c in cat_cols:
        if c in df.columns:
            m = df[c].astype(str).mode()
            cat_modes[c] = str(m.iloc[0]) if len(m) > 0 else "Unknown"
    return X, df[FRAUD_TARGET], df[AMOUNT_TARGET], encoders, list(X.columns), cat_cols, cat_modes


def _ensure_no_nan(X):
    """Replace any remaining NaN/inf with 0. Models like GradientBoostingClassifier reject NaN."""
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def run_fraud_pipeline(
    X_train: np.ndarray, X_test: np.ndarray, yf_train: pd.Series, yf_test: pd.Series
) -> tuple[ClassifierMixin | None, str | None, dict[str, dict[str, float]]]:
    """Use 5-fold CV to pick best classifier by mean F1. Train on natural (imbalanced) data; class_weight/scale_pos_weight handle imbalance."""
    X_train = _ensure_no_nan(X_train)
    X_test = _ensure_no_nan(X_test)
    n_neg = int((yf_train == 0).sum())
    n_pos = max(1, int((yf_train == 1).sum()))
    scale_pos_weight = n_neg / n_pos

    models = _fraud_models()
    best_cv_f1, best_name, best_clf = -1.0, None, None
    metrics = {}
    for name, clf in models:
        clf_fit = clone(clf)
        if "XGBoost" in name or "LightGBM" in name:
            clf_fit.set_params(scale_pos_weight=scale_pos_weight)
        try:
            cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
            cv_scores = cross_val_score(clf_fit, X_train, yf_train, cv=cv, scoring="f1", n_jobs=-1)
            mean_cv_f1 = float(np.mean(cv_scores))
        except Exception:
            mean_cv_f1 = 0.0
        clf_fit.fit(X_train, yf_train)
        pred = clf_fit.predict(X_test)
        acc = accuracy_score(yf_test, pred)
        prec = precision_score(yf_test, pred, zero_division="warn")
        rec = recall_score(yf_test, pred, zero_division="warn")
        f1 = f1_score(yf_test, pred, zero_division="warn")
        try:
            auc = roc_auc_score(yf_test, clf_fit.predict_proba(X_test)[:, 1])
        except Exception:
            auc = 0.0
        metrics[name] = {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4), "roc_auc": round(float(auc), 4), "cv_f1_mean": round(mean_cv_f1, 4)}
        if mean_cv_f1 > best_cv_f1 or (mean_cv_f1 == best_cv_f1 and rec > (metrics.get(best_name, {}).get("recall", 0) if best_name else 0)):
            best_cv_f1, best_name, best_clf = mean_cv_f1, name, clf_fit
    return best_clf, best_name, metrics


def run_severity_pipeline(
    X_train: np.ndarray, X_test: np.ndarray, ya_train: np.ndarray, ya_test: np.ndarray
) -> tuple[RegressorMixin | None, str | None, dict[str, dict[str, float]]]:
    """Use 5-fold CV to pick best regressor by mean R²; regularized models to avoid overfitting."""
    X_train = _ensure_no_nan(X_train)
    X_test = _ensure_no_nan(X_test)
    models = _severity_models()
    best_cv_r2, best_name, best_reg = -1e9, None, None
    metrics = {}
    for name, reg in models:
        try:
            cv_scores = cross_val_score(reg, X_train, ya_train, cv=N_CV, scoring="r2", n_jobs=-1)
            mean_cv_r2 = float(np.mean(cv_scores))
        except Exception:
            mean_cv_r2 = -1e9
        reg_fit = clone(reg)
        reg_fit.fit(X_train, ya_train)
        pred_log = reg_fit.predict(X_test)
        pred_dollars = np.expm1(pred_log)
        ya_test_dollars = np.expm1(ya_test)
        mae = mean_absolute_error(ya_test_dollars, pred_dollars)
        rmse = np.sqrt(mean_squared_error(ya_test_dollars, pred_dollars))
        r2 = r2_score(ya_test, pred_log)  # R2 on log scale (primary metric)
        metrics[name] = {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "r2": round(float(r2), 4), "cv_r2_mean": round(mean_cv_r2, 4)}
        if mean_cv_r2 > best_cv_r2:
            best_cv_r2, best_name, best_reg = mean_cv_r2, name, reg_fit
    return best_reg, best_name, metrics


def run_full_pipeline(csv_path: str, save_path: str | None = None) -> dict[str, Any] | None:
    """Load data, prepare features, split, run fraud and severity pipelines, return and optionally save model_store."""
    df = load_and_clean_data(csv_path)
    if df is None or len(df) < 200:
        return None
    prep = prepare_features(df)
    if prep is None:
        return None
    X, y_fraud, y_amount, encoders, feature_names, cat_cols, cat_modes = prep
    # Impute numeric features with medians; no keep_empty_features flag to avoid version-specific internals
    imputer = SimpleImputer(strategy="median", fill_value=0)
    X_imputed = imputer.fit_transform(X)
    X_imputed = _ensure_no_nan(X_imputed)
    X_train, X_test, yf_train, yf_test, ya_train, ya_test = train_test_split(
        X_imputed, y_fraud, y_amount, test_size=TEST_SIZE, random_state=SEED, stratify=y_fraud
    )
    # Explicit types for BasedPyright (train_test_split return type is loosely inferred)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    yf_train = pd.Series(yf_train, dtype=int)
    yf_test = pd.Series(yf_test, dtype=int)
    # Log-transform claim amount for severity: right-skewed amounts predict better in log space
    ya_train_arr = np.asarray(ya_train, dtype=float)
    ya_test_arr = np.asarray(ya_test, dtype=float)
    ya_train_log = np.log1p(ya_train_arr)
    ya_test_log = np.log1p(ya_test_arr)
    best_fraud, best_fraud_name, fraud_metrics = run_fraud_pipeline(X_train, X_test, yf_train, yf_test)
    best_severity, best_severity_name, severity_metrics = run_severity_pipeline(X_train, X_test, ya_train_log, ya_test_log)
    if best_fraud is None:
        return None

    ya_full_arr = np.asarray(y_amount, dtype=float)
    ya_full_log = np.log1p(ya_full_arr)

    # Retrain best models on FULL data for production (most accurate - uses every sample)
    X_full = _ensure_no_nan(X_imputed)
    yf_full = np.asarray(y_fraud, dtype=int)
    if RETRAIN_BEST_ON_FULL_DATA:
        n_neg = int((yf_full == 0).sum())
        n_pos = max(1, int((yf_full == 1).sum()))
        scale_pos_weight = n_neg / n_pos
        best_fraud_clf: Any = clone(cast(ClassifierMixin, best_fraud))
        if "scale_pos_weight" in best_fraud_clf.get_params():
            best_fraud_clf.set_params(scale_pos_weight=scale_pos_weight)
        best_fraud_clf.fit(X_full, yf_full)
        best_fraud = best_fraud_clf

        best_sev_reg: Any = clone(cast(RegressorMixin, best_severity))
        best_sev_reg.fit(X_full, ya_full_log)
        best_severity = best_sev_reg

    train_means = {c: float(v) if pd.notna(v) else 0.0 for c, v in pd.DataFrame(X_imputed, columns=feature_names).mean().items()}
    # Choose threshold that maximizes F1 on validation data (no leakage)
    fraud_threshold = 0.35
    try:
        if RETRAIN_BEST_ON_FULL_DATA:
            # Use stratified CV to find best threshold (train on natural imbalanced data)
            cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
            best_thresholds: list[float] = []
            for train_idx, val_idx in cv.split(X_full, yf_full):
                X_tr, X_val = X_full[train_idx], X_full[val_idx]
                y_tr, y_val = yf_full[train_idx], yf_full[val_idx]
                n_neg_cv, n_pos_cv = int((y_tr == 0).sum()), max(1, int((y_tr == 1).sum()))
                clf_cv: Any = clone(cast(ClassifierMixin, best_fraud))
                if "scale_pos_weight" in clf_cv.get_params():
                    clf_cv.set_params(scale_pos_weight=n_neg_cv / n_pos_cv)
                clf_cv.fit(X_tr, y_tr)
                probs = clf_cv.predict_proba(X_val)[:, 1]
                best_t, best_f1 = 0.35, 0.0
                for t in np.linspace(0.18, 0.55, 15):
                    pred_t = (probs >= t).astype(int)
                    f1_t = f1_score(y_val, pred_t, zero_division="warn")  # type: ignore[call-arg]
                    if f1_t > best_f1:
                        best_f1, best_t = f1_t, float(t)
                best_thresholds.append(best_t)
            fraud_threshold = float(np.median(best_thresholds)) if best_thresholds else 0.35
        else:
            X_test_nd = _ensure_no_nan(X_test)
            probs = cast(Any, best_fraud).predict_proba(X_test_nd)[:, 1]
            best_f1, best_rec = 0.0, 0.0
            for t in np.linspace(0.18, 0.55, 20):
                pred_t = (probs >= t).astype(int)
                f1_t = f1_score(yf_test, pred_t, zero_division="warn")
                rec_t = recall_score(yf_test, pred_t, zero_division="warn")
                if f1_t > best_f1 or (f1_t == best_f1 and rec_t > best_rec):
                    best_f1, best_rec, fraud_threshold = f1_t, rec_t, float(t)
    except Exception:
        pass
    store = {
        "fraud_model": best_fraud, "severity_model": best_severity,
        "feature_names": feature_names, "train_means": train_means,
        "imputer": imputer, "encoders": encoders, "cat_cols": cat_cols, "cat_modes": cat_modes,
        "fraud_threshold": fraud_threshold,
        "fraud_metrics": fraud_metrics, "severity_metrics": severity_metrics,
        "best_fraud_name": best_fraud_name, "best_severity_name": best_severity_name,
        "severity_log_target": True,  # model predicts log(amount); convert with expm1 at predict time
    }
    if save_path:
        joblib.dump(store, save_path)
    return store


def predict_from_store(row_dict, store):
    """Predict fraud (0/1), severity ($), and fraud probability (0–1). Uses threshold to handle imbalance."""
    feature_names = store["feature_names"]
    X_df = pd.DataFrame([row_dict]).reindex(columns=feature_names, fill_value=0).fillna(store["train_means"])
    X_arr = store["imputer"].transform(X_df)
    X_arr = _ensure_no_nan(X_arr)
    # Pass DataFrame with column names so LightGBM/XGBoost use correct feature order
    X = pd.DataFrame(X_arr, columns=feature_names)
    fraud_clf = store["fraud_model"]
    try:
        fraud_prob = float(fraud_clf.predict_proba(X)[0, 1])
    except Exception:
        fraud_prob = 0.5
    threshold = store.get("fraud_threshold", 0.5)
    fraud_class = 1 if fraud_prob >= threshold else 0
    severity_log = float(store["severity_model"].predict(X)[0])
    severity_val = np.expm1(severity_log) if store.get("severity_log_target", False) else severity_log
    return fraud_class, severity_val, fraud_prob


if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "uploads")
    csv_path = os.path.join(data_dir, "insurance_claims_fraud.csv")
    if not os.path.exists(csv_path):
        print("Place insurance_claims_fraud.csv in uploads/.")
        sys.exit(1)
    out = os.path.join(base, "fraud_severity_models.pkl")
    store = run_full_pipeline(csv_path, save_path=out)
    if store is None:
        print("Pipeline failed.")
        sys.exit(1)
    print("Best fraud:", store["best_fraud_name"], "| Best severity:", store["best_severity_name"], "| Saved:", out)
    print("\n--- Fraud models (CV F1 = model selection; Test = held-out) ---")
    for name, m in store["fraud_metrics"].items():
        cv = m.get("cv_f1_mean", "")
        cv_str = f" CV_F1={cv}" if cv != "" else ""
        print(f"{name}:{cv_str} Test Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}")
    print("\n--- Severity models (CV R² = model selection; Test = held-out) ---")
    for name, m in store["severity_metrics"].items():
        cv = m.get("cv_r2_mean", "")
        cv_str = f" CV_R2={cv}" if cv != "" else ""
        print(f"{name}:{cv_str} Test MAE=${m['mae']:.0f}, R²={m['r2']:.4f}")
