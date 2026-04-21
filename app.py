"""Insurance Claims Fraud ML — web app for fraud detection and loss severity. Uses ml_pipeline for ML."""

import os
import sys
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import subprocess

# Auto-install packages so user does not have to run pip manually
def _ensure_packages():
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req):
        print("Auto-installing packages (first run may take a minute)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req], timeout=180)
        print("Done.\n")

_ensure_packages()

import socket
import sqlite3
import hashlib
import json
import pandas as pd
import numpy as np
from urllib.parse import quote
from typing import Any, cast
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import joblib
import base64
import io

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(x: Any) -> float:
    """Convert a pandas scalar or numpy scalar to Python float for type checkers."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _is_na_scalar(v: Any) -> bool:
    """Return True if v is None or a scalar NA (e.g. np.nan). Safe for use in conditionals."""
    if v is None:
        return True
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _pip_install(name):
    subprocess.run([sys.executable, "-m", "pip", "install", name], timeout=120)

try:
    from ml_pipeline import load_and_clean_data, run_full_pipeline, predict_from_store, prepare_features, _ensure_no_nan, train_test_split, TEST_SIZE, SEED
except ModuleNotFoundError as e:
    missing = e.name
    pip_name = "scikit-learn" if missing == "sklearn" else missing
    print("Missing:", missing, "- Auto-installing", pip_name, "...")
    _pip_install(pip_name)
    try:
        from ml_pipeline import load_and_clean_data, run_full_pipeline, predict_from_store, prepare_features, _ensure_no_nan, train_test_split, TEST_SIZE, SEED
    except ModuleNotFoundError:
        print("Installing all requirements...")
        req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        if os.path.exists(req):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req], timeout=180)
        try:
            from ml_pipeline import load_and_clean_data, run_full_pipeline, predict_from_store, prepare_features, _ensure_no_nan, train_test_split, TEST_SIZE, SEED
        except ModuleNotFoundError:
            print("Install failed. Please run in this folder:  pip install -r requirements.txt")
            sys.exit(1)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'insurance-fraud-app-2024')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.template_filter('dollar')
def dollar_filter(v):
    try:
        return '$' + '{:,.2f}'.format(float(v))
    except (TypeError, ValueError):
        return '-'

@app.template_filter('fraud_yes_no')
def fraud_yes_no_filter(v):
    if v is None: return 'No'
    s = str(v).upper().strip()
    if s in ('Y', '1', 'YES', 'TRUE'): return 'Yes'
    return 'No'

DATA_FILE = 'insurance_claims_fraud.csv'
MODEL_FILE = 'fraud_severity_models.pkl'
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
DATA_PATH = os.path.join(UPLOADS_DIR, 'insurance_claims_fraud.csv')
MAPPING_FILE = os.path.join(UPLOADS_DIR, 'column_mapping.json')
TEMP_UPLOAD = os.path.join(UPLOADS_DIR, 'upload_temp')
FRAUD_TARGET = 'fraud_reported'
AMOUNT_TARGET = 'total_claim_amount'
model_store = None

# Map form dropdown values to dataset values (some CSVs use "Front" not "Front Collision")
FORM_TO_DATA_COLLISION = {"Front Collision": "Front", "Rear Collision": "Rear", "Side Collision": "Side", "Rollover": "Rollover"}
FORM_TO_DATA_SEVERITY = {"Trivial Damage": "Minor Damage"}
FRAUD_ALIASES = ['fraud_reported', 'fraud', 'is_fraud', 'fraud_flag', 'fraudulent', 'claim_fraud', 'fraud_indicator']
AMOUNT_ALIASES = ['total_claim_amount', 'claim_amount', 'amount', 'total_amount', 'claim_total', 'claim_value', 'total_claim']


def _auto_detect_mapping(columns):
    """Return {FRAUD_TARGET: col, AMOUNT_TARGET: col} or None if both not found."""
    fraud_col = amount_col = None
    for c in columns:
        cl = c.lower().strip()
        if fraud_col is None and any(a in cl or cl == a for a in [a.lower() for a in FRAUD_ALIASES]):
            fraud_col = c
        if amount_col is None and any(a in cl or cl == a for a in [a.lower() for a in AMOUNT_ALIASES]):
            amount_col = c
    if fraud_col and amount_col and fraud_col != amount_col:
        return {FRAUD_TARGET: fraud_col, AMOUNT_TARGET: amount_col}
    return None


def _load_mapping():
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_mapping(mapping):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)


def _read_uploaded_file(path):
    """Load CSV or Excel into DataFrame."""
    path_l = path.lower()
    if path_l.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path).replace("?", np.nan)
    return pd.read_csv(path).replace("?", np.nan)


def _apply_mapping_and_save(df, mapping, out_path):
    """Rename columns per mapping and save to CSV."""
    reverse = {v: k for k, v in mapping.items()}
    df = df.rename(columns=reverse)
    df.to_csv(out_path, index=False)
    return df


def get_csv_path():
    """Training and prediction use only insurance_claims_fraud.csv in uploads."""
    return os.path.join(UPLOADS_DIR, "insurance_claims_fraud.csv")


def get_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'insurance_app.db'))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    if conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', ('admin', hashlib.sha256('admin123'.encode()).hexdigest()))
        conn.commit()
    conn.close()


def check_user(username, password):
    conn = get_db()
    row = conn.execute('SELECT id, username FROM users WHERE username = ? AND password_hash = ?', (username, hashlib.sha256(password.encode()).hexdigest())).fetchone()
    conn.close()
    return dict(row) if row else None


def add_user(username, password):
    conn = get_db()
    try:
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashlib.sha256(password.encode()).hexdigest()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def load_claims_data():
    path = get_csv_path()
    if not os.path.exists(path):
        return None
    df = load_and_clean_data(path)
    return df


def get_claim_stats():
    df = load_claims_data()
    if df is None or len(df) == 0 or FRAUD_TARGET not in df.columns or AMOUNT_TARGET not in df.columns:
        return None
    fraud_col = df[FRAUD_TARGET].astype(str).str.upper().str.strip()
    return {
        'total_claims': len(df), 'fraud_count': int((fraud_col == 'Y').sum()),
        'legit_count': int(len(df) - (fraud_col == 'Y').sum()),
        'avg_claim_amount': round(_to_float(df[AMOUNT_TARGET].mean()), 2),
        'min_claim_amount': round(_to_float(df[AMOUNT_TARGET].min()), 2),
        'max_claim_amount': round(_to_float(df[AMOUNT_TARGET].max()), 2),
    }


def get_models():
    global model_store
    if model_store is not None:
        return model_store
    # Prefer models trained by train_and_save_models.py (saved in trained_models/)
    trained_dir = os.path.join(BASE_DIR, 'trained_models')
    full_store_path = os.path.join(trained_dir, 'full_store.pkl')
    if os.path.exists(full_store_path):
        try:
            model_store = joblib.load(full_store_path)
            return model_store
        except Exception:
            pass
    pkl_path = os.path.join(BASE_DIR, MODEL_FILE)
    if os.path.exists(pkl_path):
        try:
            model_store = joblib.load(pkl_path)
            return model_store
        except Exception:
            pass
    if not os.path.exists(get_csv_path()):
        return None
    model_store = run_full_pipeline(get_csv_path(), save_path=pkl_path)
    return model_store


def _normalize_fraud(df):
    """Normalize fraud_reported column to Y/N for consistent display in templates."""
    if df is None or FRAUD_TARGET not in df.columns:
        return df
    d = df.copy()
    def to_yn(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N'
        s = str(v).upper().strip()
        if s in ('Y', '1', 'YES', 'TRUE'):
            return 'Y'
        return 'N'
    d[FRAUD_TARGET] = d[FRAUD_TARGET].apply(to_yn)
    return d


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_share_link():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = None
    return dict(share_link=f'http://{ip}:5000' if ip else None)


@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user' in session else url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    init_db()
    if request.method == 'POST':
        u = check_user(request.form.get('username', '').strip(), request.form.get('password', ''))
        if u:
            session['user'] = u['username']
            session['user_id'] = u['id']
            return redirect(url_for('dashboard'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password required.'
        elif len(password) < 4:
            error = 'Password at least 4 characters.'
        elif add_user(username, password):
            return redirect(url_for('login'))
        else:
            error = 'Username already exists.'
    return render_template('register.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_claim_stats() or {'total_claims': 0, 'fraud_count': 0, 'legit_count': 0, 'avg_claim_amount': 0, 'min_claim_amount': 0, 'max_claim_amount': 0}
    store = get_models()
    sample_claims = []
    df = load_claims_data()
    if df is not None and len(df) > 0:
        df = _normalize_fraud(df)
        n = min(10, len(df))
        for idx in df.index[:n]:
            row = df.loc[idx].to_dict()
            row['_idx'] = int(idx)
            sample_claims.append(row)
    return render_template('dashboard.html', stats=stats, sample_claims=sample_claims, data_ok=get_claim_stats() is not None, msg=request.args.get('msg', ''),
        fraud_metrics=store.get('fraud_metrics', {}) if store else {}, severity_metrics=store.get('severity_metrics', {}) if store else {},
        best_fraud_name=store.get('best_fraud_name', '') if store else '', best_severity_name=store.get('best_severity_name', '') if store else '',
        data_path=get_csv_path() if store else None,
        fraud_recall=store.get('fraud_metrics', {}).get(store.get('best_fraud_name'), {}).get('recall') if store and store.get('best_fraud_name') else None,
        username=session.get('user'))


def _clear_models():
    global model_store
    model_store = None
    pkl = os.path.join(BASE_DIR, MODEL_FILE)
    if os.path.exists(pkl):
        try:
            os.remove(pkl)
        except Exception:
            pass


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_dataset():
    if request.method == 'GET':
        session.pop('upload_columns', None)
        temp = session.pop('upload_temp_path', None)
        if temp and os.path.exists(temp):
            try:
                os.remove(temp)
            except Exception:
                pass
    error = None
    if request.method == 'POST' and 'dataset' in request.files:
        f = request.files['dataset']
        ext = f.filename.rsplit('.', 1)[-1].lower() if f.filename else ''
        if f.filename and ext in ('csv', 'xlsx', 'xls'):
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            temp_path = TEMP_UPLOAD + '.' + ('xlsx' if ext in ('xlsx', 'xls') else 'csv')
            f.save(temp_path)
            try:
                df = _read_uploaded_file(temp_path)
                if df is None or len(df) < 200:
                    error = 'File needs at least 200 rows for training.'
                else:
                    mapping = _auto_detect_mapping(df.columns.tolist())
                    if mapping:
                        _apply_mapping_and_save(df, mapping, DATA_PATH)
                        _save_mapping(mapping)
                        _clear_models()
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                        return redirect(url_for('dashboard', msg='Dataset uploaded. Models will retrain on next use.'))
                    else:
                        session['upload_columns'] = df.columns.tolist()
                        session['upload_temp_path'] = temp_path
                        return redirect(url_for('upload_map'))
            except Exception as e:
                error = f'Could not read file: {str(e)}'
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
        else:
            error = 'Select a CSV or Excel (.xlsx, .xls) file with vehicle claims data.'
    elif request.method == 'POST':
        error = 'No file selected.'
    return render_template('upload.html', error=error, username=session.get('user'), has_data=os.path.exists(get_csv_path()))


@app.route('/upload/map', methods=['GET', 'POST'])
@login_required
def upload_map():
    columns = session.get('upload_columns')
    temp_path = session.get('upload_temp_path')
    if not columns or not temp_path or not os.path.exists(temp_path):
        return redirect(url_for('upload_dataset'))
    error = None
    if request.method == 'POST':
        fraud_col = request.form.get('fraud_column', '').strip()
        amount_col = request.form.get('amount_column', '').strip()
        if not fraud_col or not amount_col:
            error = 'Please select both columns.'
        elif fraud_col == amount_col:
            error = 'Fraud and amount must be different columns.'
        elif fraud_col not in columns or amount_col not in columns:
            error = 'Invalid column selection.'
        else:
            try:
                df = _read_uploaded_file(temp_path)
                mapping = {FRAUD_TARGET: fraud_col, AMOUNT_TARGET: amount_col}
                _apply_mapping_and_save(df, mapping, DATA_PATH)
                _save_mapping(mapping)
                _clear_models()
                session.pop('upload_columns', None)
                session.pop('upload_temp_path', None)
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                return redirect(url_for('dashboard', msg='Dataset uploaded. Models will retrain on next use.'))
            except Exception as e:
                error = str(e)
    return render_template('upload_map.html', columns=columns, error=error, username=session.get('user'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    fraud_result = severity_result = error = fraud_why = fraud_prob = risk_level = recommendation = None
    store = get_models()
    form_used = {}
    if store is None:
        error = 'Upload a CSV or Excel file with vehicle claims (Upload page), or place insurance_claims_fraud.csv in the project folder.'
    elif request.method == 'POST':
        try:
            df = load_claims_data()
            if df is None:
                raise ValueError("No claims data loaded. Upload a dataset first.")
            feature_names = store['feature_names']
            encoders = store.get('encoders', {})
            train_means = store['train_means']

            def _default_val(c):
                if df[c].dtype in [np.int64, np.float64]:
                    return _to_float(df[c].median())
                m = df[c].mode()
                v = m.iloc[0] if len(m) > 0 else 'Unknown'
                return str(v) if pd.notna(v) else 'Unknown'

            row_raw = {c: _default_val(c) for c in feature_names if c in df.columns}
            # Map form field names to possible dataset column names (e.g. uploads use insured_age not age)
            form_to_cols = [
                ('age', int, ['age', 'insured_age']),
                ('incident_type', str, ['incident_type']),
                ('collision_type', str, ['collision_type']),
                ('incident_severity', str, ['incident_severity']),
                ('number_of_vehicles_involved', int, ['number_of_vehicles_involved']),
                ('bodily_injuries', int, ['bodily_injuries']),
            ]
            for form_key, conv, col_candidates in form_to_cols:
                form_val = request.form.get(form_key)
                if form_val is None or form_val == '':
                    continue
                try:
                    value = conv(form_val) if conv != str else form_val
                except (TypeError, ValueError):
                    continue
                # Normalize categoricals to match training data vocab (e.g. "Front Collision" -> "Front")
                if form_key == "collision_type" and isinstance(value, str):
                    value = FORM_TO_DATA_COLLISION.get(value, value)
                if form_key == "incident_severity" and isinstance(value, str):
                    value = FORM_TO_DATA_SEVERITY.get(value, value)
                for col in col_candidates:
                    if col in row_raw:
                        row_raw[col] = value
                        break

            inc_type = str(row_raw.get('incident_type', '')).lower()
            inc_sev = str(row_raw.get('incident_severity', '')).lower()
            coll = str(row_raw.get('collision_type', '')).lower()
            parked = 'parked' in inc_type
            total_loss = 'total' in inc_sev
            rollover = 'rollover' in coll
            if 'inconsistent_claim' in feature_names:
                row_raw['inconsistent_claim'] = 1 if (parked and (total_loss or rollover)) else 0
            if 'severity_total_loss' in feature_names:
                row_raw['severity_total_loss'] = 1 if total_loss else 0
            if 'multi_vehicle' in feature_names:
                try:
                    nv = int(row_raw.get('number_of_vehicles_involved', 0))
                    row_raw['multi_vehicle'] = 1 if nv >= 2 else 0
                except (TypeError, ValueError):
                    row_raw['multi_vehicle'] = 0

            bodily = row_raw.get('bodily_injuries')
            if bodily is not None and 'injury_claim' in row_raw:
                try:
                    row_raw['injury_claim'] = float(bodily) * _to_float(df['injury_claim'].median()) * 0.5 + _to_float(df['injury_claim'].median()) * 0.5
                except Exception:
                    pass
            sev = str(row_raw.get('incident_severity', '')).lower()
            if 'property_claim' in row_raw:
                try:
                    mult = 1.5 if 'total' in sev else (1.2 if 'major' in sev else (0.8 if 'minor' in sev else 0.5))
                    row_raw['property_claim'] = _to_float(df['property_claim'].median()) * mult
                except Exception:
                    pass
            if 'vehicle_claim' in row_raw:
                try:
                    mult = 1.6 if 'total' in sev else (1.2 if 'major' in sev else (0.9 if 'minor' in sev else 0.5))
                    row_raw['vehicle_claim'] = _to_float(df['vehicle_claim'].median()) * mult
                except Exception:
                    pass

            row_numeric = {}
            cat_modes = store.get("cat_modes", {})
            for c in feature_names:
                val = row_raw.get(c)
                if c in encoders:
                    val_str = str(val) if not _is_na_scalar(val) else 'Unknown'
                    le = encoders[c]
                    classes = list(le.classes_)
                    if val_str not in classes:
                        val_str = str(cat_modes.get(c, classes[0] if len(classes) > 0 else 'Unknown'))
                    if val_str not in classes:
                        val_str = classes[0] if len(classes) > 0 else 'Unknown'
                    row_numeric[c] = float(le.transform([val_str])[0])
                elif val is not None and not _is_na_scalar(val):
                    try:
                        row_numeric[c] = float(val)
                    except (TypeError, ValueError):
                        row_numeric[c] = float(train_means.get(c, 0))
                else:
                    row_numeric[c] = float(train_means.get(c, 0))

            fraud_result, severity_result, fraud_prob = predict_from_store(row_numeric, store)
            severity_result = round(float(severity_result), 2)
            # Production: risk level and clear recommendation for users
            if fraud_prob is not None:
                if fraud_prob < 0.35:
                    risk_level, recommendation = "Low", "No action recommended"
                elif fraud_prob < 0.65:
                    risk_level, recommendation = "Medium", "Consider manual review"
                else:
                    risk_level, recommendation = "High", "Flag for review"
            else:
                risk_level, recommendation = None, None
            fraud_why = (
                "The model predicted Yes (fraud) because the claim details you entered (incident type, severity, age, vehicles involved, etc.) match patterns that in the training data are often associated with fraudulent claims. Such claims may be flagged for review."
                if fraud_result == 1 else
                "The model predicted No (not fraud) because the claim details you entered match patterns that in the training data are typically associated with legitimate claims. No extra review is suggested based on this prediction."
            )
        except Exception as e:
            error = str(e)
            fraud_why = None
    else:
        fraud_why = None
    if request.method == 'POST' and request.form:
        for k in ['age', 'incident_type', 'collision_type', 'incident_severity', 'number_of_vehicles_involved', 'bodily_injuries']:
            form_used[k] = request.form.get(k, '')
    error = error or request.args.get('error')
    return render_template('predict.html', fraud_result=fraud_result, severity_result=severity_result, fraud_prob=fraud_prob, risk_level=risk_level, recommendation=recommendation, fraud_why=fraud_why, error=error, username=session.get('user'), form_used=form_used)


@app.route('/predict/severity', methods=['GET', 'POST'])
@login_required
def predict_severity():
    severity_result = error = None
    store = get_models()
    form_used: dict[str, Any] = {}
    if store is None:
        error = 'Upload a CSV or Excel file with vehicle claims (Upload page), or place insurance_claims_fraud.csv in the project folder.'
    elif request.method == 'POST':
        try:
            df = load_claims_data()
            if df is None:
                raise ValueError("No claims data loaded. Upload a dataset first.")
            feature_names = store['feature_names']
            encoders = store.get('encoders', {})
            train_means = store['train_means']

            def _default_val(c):
                if df[c].dtype in [np.int64, np.float64]:
                    return _to_float(df[c].median())
                m = df[c].mode()
                v = m.iloc[0] if len(m) > 0 else 'Unknown'
                return str(v) if pd.notna(v) else 'Unknown'

            row_raw = {c: _default_val(c) for c in feature_names if c in df.columns}
            form_to_cols = [
                ('age', int, ['age', 'insured_age']),
                ('incident_type', str, ['incident_type']),
                ('collision_type', str, ['collision_type']),
                ('incident_severity', str, ['incident_severity']),
                ('number_of_vehicles_involved', int, ['number_of_vehicles_involved']),
                ('bodily_injuries', int, ['bodily_injuries']),
            ]
            for form_key, conv, col_candidates in form_to_cols:
                form_val = request.form.get(form_key)
                if form_val is None or form_val == '':
                    continue
                try:
                    value = conv(form_val) if conv != str else form_val
                except (TypeError, ValueError):
                    continue
                if form_key == "collision_type" and isinstance(value, str):
                    value = FORM_TO_DATA_COLLISION.get(value, value)
                if form_key == "incident_severity" and isinstance(value, str):
                    value = FORM_TO_DATA_SEVERITY.get(value, value)
                for col in col_candidates:
                    if col in row_raw:
                        row_raw[col] = value
                        break

            inc_type = str(row_raw.get('incident_type', '')).lower()
            inc_sev = str(row_raw.get('incident_severity', '')).lower()
            coll = str(row_raw.get('collision_type', '')).lower()
            parked = 'parked' in inc_type
            total_loss = 'total' in inc_sev
            rollover = 'rollover' in coll
            if 'inconsistent_claim' in feature_names:
                row_raw['inconsistent_claim'] = 1 if (parked and (total_loss or rollover)) else 0
            if 'severity_total_loss' in feature_names:
                row_raw['severity_total_loss'] = 1 if total_loss else 0
            if 'multi_vehicle' in feature_names:
                try:
                    nv = int(row_raw.get('number_of_vehicles_involved', 0))
                    row_raw['multi_vehicle'] = 1 if nv >= 2 else 0
                except (TypeError, ValueError):
                    row_raw['multi_vehicle'] = 0

            bodily = row_raw.get('bodily_injuries')
            if bodily is not None and 'injury_claim' in row_raw:
                try:
                    row_raw['injury_claim'] = float(bodily) * _to_float(df['injury_claim'].median()) * 0.5 + _to_float(df['injury_claim'].median()) * 0.5
                except Exception:
                    pass
            sev = str(row_raw.get('incident_severity', '')).lower()
            if 'property_claim' in row_raw:
                try:
                    mult = 1.5 if 'total' in sev else (1.2 if 'major' in sev else (0.8 if 'minor' in sev else 0.5))
                    row_raw['property_claim'] = _to_float(df['property_claim'].median()) * mult
                except Exception:
                    pass
            if 'vehicle_claim' in row_raw:
                try:
                    mult = 1.6 if 'total' in sev else (1.2 if 'major' in sev else (0.9 if 'minor' in sev else 0.5))
                    row_raw['vehicle_claim'] = _to_float(df['vehicle_claim'].median()) * mult
                except Exception:
                    pass

            row_numeric: dict[str, float] = {}
            cat_modes = store.get("cat_modes", {})
            for c in feature_names:
                val = row_raw.get(c)
                if c in encoders:
                    val_str = str(val) if not _is_na_scalar(val) else 'Unknown'
                    le = encoders[c]
                    classes = list(le.classes_)
                    if val_str not in classes:
                        val_str = str(cat_modes.get(c, classes[0] if len(classes) > 0 else 'Unknown'))
                    if val_str not in classes:
                        val_str = classes[0] if len(classes) > 0 else 'Unknown'
                    row_numeric[c] = float(le.transform([val_str])[0])
                elif val is not None and not _is_na_scalar(val):
                    try:
                        row_numeric[c] = float(val)
                    except (TypeError, ValueError):
                        row_numeric[c] = float(train_means.get(c, 0))
                else:
                    row_numeric[c] = float(train_means.get(c, 0))

            _, severity_val, _ = predict_from_store(row_numeric, store)
            severity_result = round(float(severity_val), 2)
        except Exception as e:
            error = str(e)
    if request.method == 'POST' and request.form:
        for k in ['age', 'incident_type', 'collision_type', 'incident_severity', 'number_of_vehicles_involved', 'bodily_injuries']:
            form_used[k] = request.form.get(k, '')
    error = error or request.args.get('error')
    return render_template('predict_severity.html', severity_result=severity_result, error=error, username=session.get('user'), form_used=form_used)


@app.route('/evaluation')
@login_required
def evaluation():
    """Evaluation page: confusion matrix, ROC curve, precision-recall curve, and fraud metrics (F1, precision, recall, accuracy)."""
    store = get_models()
    if store is None:
        return redirect(url_for('dashboard') + '?error=' + quote('No model loaded. Train or load models first.'))
    path = get_csv_path()
    if not path or not os.path.exists(path):
        return render_template('evaluation.html', error='No dataset found. Upload insurance_claims_fraud.csv first.', username=session.get('user'))
    df = load_and_clean_data(path)
    if df is None or len(df) < 100:
        return render_template('evaluation.html', error='Dataset too small or could not be loaded.', username=session.get('user'))
    prep = prepare_features(df)
    if prep is None:
        return render_template('evaluation.html', error='Feature preparation failed (e.g. fraud column has only one class).', username=session.get('user'))
    X, y_fraud, y_amount, encoders, feature_names, cat_cols, cat_modes = prep
    X_imputed = store["imputer"].transform(X)
    X_imputed = _ensure_no_nan(X_imputed)
    X_train, X_test, yf_train, yf_test = train_test_split(
        X_imputed, y_fraud, test_size=TEST_SIZE, random_state=SEED, stratify=y_fraud
    )
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    try:
        y_prob = store["fraud_model"].predict_proba(X_test_df)[:, 1]
    except Exception:
        return render_template('evaluation.html', error='Could not get fraud model probabilities.', username=session.get('user'))
    threshold = store.get("fraud_threshold", 0.5)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(yf_test, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))  # type: ignore[arg-type]
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # type: ignore[arg-type]
    f1 = float(f1_score(y_true, y_pred, zero_division=0))  # type: ignore[arg-type]
    roc_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    avg_prec = 0.0
    try:
        avg_prec = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plt.figure(figsize=(5, 4))
    ax_cm = fig_cm.gca()
    im = ax_cm.imshow(cm, cmap='Blues')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Not Fraud', 'Fraud'])
    ax_cm.set_yticklabels(['Not Fraud', 'Fraud'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontweight='bold')
    plt.colorbar(im, ax=ax_cm, label='Count')
    plt.tight_layout()
    buf_cm = io.BytesIO()
    fig_cm.savefig(buf_cm, format='png', dpi=120, bbox_inches='tight')
    buf_cm.seek(0)
    img_cm = base64.b64encode(buf_cm.read()).decode('utf-8')
    plt.close(fig_cm)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig_roc = plt.figure(figsize=(5, 4))
    ax_roc = fig_roc.gca()
    ax_roc.plot(fpr, tpr, color='#0d9488', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    buf_roc = io.BytesIO()
    fig_roc.savefig(buf_roc, format='png', dpi=120, bbox_inches='tight')
    buf_roc.seek(0)
    img_roc = base64.b64encode(buf_roc.read()).decode('utf-8')
    plt.close(fig_roc)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    fig_pr = plt.figure(figsize=(5, 4))
    ax_pr = fig_pr.gca()
    ax_pr.plot(rec_curve, prec_curve, color='#0d9488', lw=2, label=f'PR (AP = {avg_prec:.3f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision–Recall Curve')
    ax_pr.legend(loc='upper right')
    ax_pr.grid(True, alpha=0.3)
    plt.tight_layout()
    buf_pr = io.BytesIO()
    fig_pr.savefig(buf_pr, format='png', dpi=120, bbox_inches='tight')
    buf_pr.seek(0)
    img_pr = base64.b64encode(buf_pr.read()).decode('utf-8')
    plt.close(fig_pr)

    return render_template(
        'evaluation.html',
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        avg_precision=avg_prec,
        n_test=len(y_true),
        img_confusion_matrix=img_cm,
        img_roc=img_roc,
        img_pr=img_pr,
        best_fraud_name=store.get('best_fraud_name', 'Fraud model'),
        best_severity_name=store.get('best_severity_name', ''),
        fraud_metrics=store.get('fraud_metrics', {}),
        severity_metrics=store.get('severity_metrics', {}),
        username=session.get('user'),
    )


def _payload_to_row_numeric(payload, df, store):
    """Build row_numeric dict from API JSON payload. Uses df for defaults; same feature logic as web predict."""
    feature_names = store["feature_names"]
    encoders = store.get("encoders", {})
    train_means = store["train_means"]
    cat_modes = store.get("cat_modes", {})

    def _default_val(c):
        if c not in df.columns:
            return 0.0 if c in train_means else "Unknown"
        if df[c].dtype in [np.int64, np.float64]:
            return _to_float(df[c].median())
        m = df[c].mode()
        v = m.iloc[0] if len(m) > 0 else "Unknown"
        return str(v) if pd.notna(v) else "Unknown"

    row_raw = {c: _default_val(c) for c in feature_names if c in df.columns}
    form_to_cols = [
        ("age", int, ["age", "insured_age"]),
        ("incident_type", str, ["incident_type"]),
        ("collision_type", str, ["collision_type"]),
        ("incident_severity", str, ["incident_severity"]),
        ("number_of_vehicles_involved", int, ["number_of_vehicles_involved"]),
        ("bodily_injuries", int, ["bodily_injuries"]),
    ]
    for form_key, conv, col_candidates in form_to_cols:
        val = payload.get(form_key)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        try:
            value = conv(val) if conv != str else str(val).strip()
        except (TypeError, ValueError):
            continue
        if form_key == "collision_type" and isinstance(value, str):
            value = FORM_TO_DATA_COLLISION.get(value, value)
        if form_key == "incident_severity" and isinstance(value, str):
            value = FORM_TO_DATA_SEVERITY.get(value, value)
        for col in col_candidates:
            if col in row_raw:
                row_raw[col] = value
                break

    inc_type = str(row_raw.get("incident_type", "")).lower()
    inc_sev = str(row_raw.get("incident_severity", "")).lower()
    coll = str(row_raw.get("collision_type", "")).lower()
    parked = "parked" in inc_type
    total_loss = "total" in inc_sev
    rollover = "rollover" in coll
    if "inconsistent_claim" in feature_names:
        row_raw["inconsistent_claim"] = 1 if (parked and (total_loss or rollover)) else 0
    if "severity_total_loss" in feature_names:
        row_raw["severity_total_loss"] = 1 if total_loss else 0
    if "multi_vehicle" in feature_names:
        try:
            nv = int(row_raw.get("number_of_vehicles_involved", 0))
            row_raw["multi_vehicle"] = 1 if nv >= 2 else 0
        except (TypeError, ValueError):
            row_raw["multi_vehicle"] = 0

    bodily = row_raw.get("bodily_injuries")
    if bodily is not None and "injury_claim" in row_raw:
        try:
            row_raw["injury_claim"] = float(bodily) * _to_float(df["injury_claim"].median()) * 0.5 + _to_float(df["injury_claim"].median()) * 0.5
        except Exception:
            pass
    sev = str(row_raw.get("incident_severity", "")).lower()
    if "property_claim" in row_raw:
        try:
            mult = 1.5 if "total" in sev else (1.2 if "major" in sev else (0.8 if "minor" in sev else 0.5))
            row_raw["property_claim"] = _to_float(df["property_claim"].median()) * mult
        except Exception:
            pass
    if "vehicle_claim" in row_raw:
        try:
            mult = 1.6 if "total" in sev else (1.2 if "major" in sev else (0.9 if "minor" in sev else 0.5))
            row_raw["vehicle_claim"] = _to_float(df["vehicle_claim"].median()) * mult
        except Exception:
            pass

    row_numeric = {}
    for c in feature_names:
        val = row_raw.get(c)
        if c in encoders:
            val_str = str(val) if not _is_na_scalar(val) else "Unknown"
            le = encoders[c]
            classes = list(le.classes_)
            if val_str not in classes:
                val_str = str(cat_modes.get(c, classes[0] if len(classes) > 0 else "Unknown"))
            if val_str not in classes:
                val_str = classes[0] if len(classes) > 0 else "Unknown"
            row_numeric[c] = float(le.transform([val_str])[0])
        elif val is not None and not _is_na_scalar(val):
            try:
                row_numeric[c] = float(val)
            except (TypeError, ValueError):
                row_numeric[c] = float(train_means.get(c, 0))
        else:
            row_numeric[c] = float(train_means.get(c, 0))
    return row_numeric


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API: POST JSON with claim fields; returns fraud flag, fraud probability, and predicted amount.
    No auth required for easy demo. Use in production with API key or auth if needed.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    store = get_models()
    if store is None:
        return jsonify({"error": "No model loaded. Train or load models first (e.g. run app and ensure trained_models/full_store.pkl exists)."}), 503
    df = load_claims_data()
    if df is None:
        return jsonify({"error": "No training data. Upload insurance_claims_fraud.csv or use Upload page."}), 503

    try:
        row_numeric = _payload_to_row_numeric(payload, df, store)
        fraud_class, severity_val, fraud_prob = predict_from_store(row_numeric, store)
        return jsonify({
            "fraud": bool(fraud_class),
            "fraud_probability": round(float(fraud_prob), 4),
            "predicted_amount": round(float(severity_val), 2),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _build_row_numeric_from_series(row_series, df_train, store):
    """Build row_numeric dict from a single CSV row (Series) for batch predict. Uses df_train for defaults."""
    feature_names = store["feature_names"]
    encoders = store.get("encoders", {})
    train_means = store["train_means"]
    cat_modes = store.get("cat_modes", {})
    # Map batch CSV column names to feature names (batch may have age vs insured_age etc.)
    col_aliases = {"age": ["age", "Age", "insured_age"], "insured_age": ["insured_age", "age", "Age"]}

    def _get_val(c):
        if c in row_series.index and pd.notna(row_series.get(c)):
            return row_series[c]
        for alias in col_aliases.get(c, [c]):
            if alias in row_series.index and pd.notna(row_series.get(alias)):
                return row_series[alias]
        return None

    def _default_val(c):
        if c not in df_train.columns:
            return 0.0 if c in train_means else "Unknown"
        if df_train[c].dtype in [np.int64, np.float64]:
            return _to_float(df_train[c].median())
        m = df_train[c].mode()
        v = m.iloc[0] if len(m) > 0 else "Unknown"
        return str(v) if pd.notna(v) else "Unknown"

    row_raw = {}
    for c in feature_names:
        v = _get_val(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row_raw[c] = _default_val(c)
        else:
            row_raw[c] = v

    # Normalize categoricals from batch (e.g. "Front Collision" -> "Front")
    if "collision_type" in row_raw and isinstance(row_raw["collision_type"], str):
        row_raw["collision_type"] = FORM_TO_DATA_COLLISION.get(row_raw["collision_type"], row_raw["collision_type"])
    if "incident_severity" in row_raw and isinstance(row_raw["incident_severity"], str):
        row_raw["incident_severity"] = FORM_TO_DATA_SEVERITY.get(row_raw["incident_severity"], row_raw["incident_severity"])

    inc_type = str(row_raw.get("incident_type", "")).lower()
    inc_sev = str(row_raw.get("incident_severity", "")).lower()
    coll = str(row_raw.get("collision_type", "")).lower()
    parked = "parked" in inc_type
    total_loss = "total" in inc_sev
    rollover = "rollover" in coll
    if "inconsistent_claim" in feature_names:
        row_raw["inconsistent_claim"] = 1 if (parked and (total_loss or rollover)) else 0
    if "severity_total_loss" in feature_names:
        row_raw["severity_total_loss"] = 1 if total_loss else 0
    if "multi_vehicle" in feature_names:
        try:
            nv = int(row_raw.get("number_of_vehicles_involved", 0))
            row_raw["multi_vehicle"] = 1 if nv >= 2 else 0
        except (TypeError, ValueError):
            row_raw["multi_vehicle"] = 0

    # Derived claim amounts from severity / bodily_injuries if missing
    sev = str(row_raw.get("incident_severity", "")).lower()
    if "injury_claim" in row_raw and _is_na_scalar(row_raw.get("injury_claim")):
        try:
            bodily = row_raw.get("bodily_injuries", 0)
            bodily = float(bodily) if bodily is not None else 0
            row_raw["injury_claim"] = bodily * _to_float(df_train["injury_claim"].median()) * 0.5 + _to_float(df_train["injury_claim"].median()) * 0.5
        except Exception:
            row_raw["injury_claim"] = _to_float(df_train["injury_claim"].median()) if "injury_claim" in df_train.columns else 0
    if "property_claim" in row_raw and _is_na_scalar(row_raw.get("property_claim")):
        try:
            mult = 1.5 if "total" in sev else (1.2 if "major" in sev else (0.8 if "minor" in sev else 0.5))
            row_raw["property_claim"] = _to_float(df_train["property_claim"].median()) * mult
        except Exception:
            row_raw["property_claim"] = float(train_means.get("property_claim", 0))
    if "vehicle_claim" in row_raw and _is_na_scalar(row_raw.get("vehicle_claim")):
        try:
            mult = 1.6 if "total" in sev else (1.2 if "major" in sev else (0.9 if "minor" in sev else 0.5))
            row_raw["vehicle_claim"] = _to_float(df_train["vehicle_claim"].median()) * mult
        except Exception:
            row_raw["vehicle_claim"] = float(train_means.get("vehicle_claim", 0))

    row_numeric = {}
    for c in feature_names:
        val = row_raw.get(c)
        if c in encoders:
            val_str = str(val) if not _is_na_scalar(val) else "Unknown"
            le = encoders[c]
            classes = list(le.classes_)
            if val_str not in classes:
                val_str = str(cat_modes.get(c, classes[0] if len(classes) > 0 else "Unknown"))
            if val_str not in classes:
                val_str = classes[0] if len(classes) > 0 else "Unknown"
            row_numeric[c] = float(le.transform([val_str])[0])
        elif val is not None and not _is_na_scalar(val):
            try:
                row_numeric[c] = float(val)
            except (TypeError, ValueError):
                row_numeric[c] = float(train_means.get(c, 0))
        else:
            row_numeric[c] = float(train_means.get(c, 0))
    return row_numeric


@app.route('/predict/batch', methods=['POST'])
@login_required
def predict_batch():
    """Upload a CSV of claims; return a new CSV with columns: fraud_predicted (Y/N), fraud_probability, predicted_amount, risk_level."""
    store = get_models()
    if store is None:
        return redirect(url_for('predict') + '?error=' + quote('No model loaded. Upload a dataset first (Upload page).'))
    f = request.files.get('batch_csv')
    if not f or not f.filename or not f.filename.lower().endswith('.csv'):
        return redirect(url_for('predict') + '?error=' + quote('Please upload a CSV file.'))
    try:
        df_batch = pd.read_csv(f.stream).replace("?", np.nan)
    except Exception as e:
        return redirect(url_for('predict') + '?error=' + quote('Could not read CSV: ' + str(e)))
    if len(df_batch) == 0:
        return redirect(url_for('predict') + '?error=' + quote('CSV has no rows.'))
    df_train = load_claims_data()
    if df_train is None:
        return redirect(url_for('predict') + '?error=' + quote('Training data not found. Use Upload page first.'))

    fraud_pred = []
    fraud_prob_list = []
    pred_amount_list = []
    risk_list = []
    for idx in df_batch.index:
        row_numeric = _build_row_numeric_from_series(df_batch.loc[idx], df_train, store)
        fraud_class, severity_val, fraud_prob = predict_from_store(row_numeric, store)
        fraud_pred.append('Y' if fraud_class == 1 else 'N')
        fraud_prob_list.append(round(float(fraud_prob), 4))
        pred_amount_list.append(round(float(severity_val), 2))
        if fraud_prob < 0.35:
            risk_list.append('Low')
        elif fraud_prob < 0.65:
            risk_list.append('Medium')
        else:
            risk_list.append('High')
    df_batch = df_batch.copy()
    df_batch['fraud_predicted'] = fraud_pred
    df_batch['fraud_probability'] = fraud_prob_list
    df_batch['predicted_amount'] = pred_amount_list
    df_batch['risk_level'] = risk_list

    import io
    buf = io.BytesIO()
    df_batch.to_csv(buf, index=False, encoding='utf-8')
    buf.seek(0)
    out_name = 'claims_with_fraud_predictions.csv'
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name=out_name)
    if df is None or FRAUD_TARGET not in df.columns:
        return df
    d = df.copy()
    d[FRAUD_TARGET] = d[FRAUD_TARGET].astype(str).str.upper().str.strip().replace({'1': 'Y', '0': 'N'})
    return d


@app.route('/claims')
@login_required
def claims_list():
    df = load_claims_data()
    if df is None:
        return render_template('claims.html', data=[], page=1, total_pages=0, total=0, search='', sort='total_claim_amount', order='desc', username=session.get('user'))
    df = _normalize_fraud(df)
    search = (request.args.get('q') or '').strip()
    if search:
        mask = pd.Series(False, index=df.index)
        for col in ['policy_number', 'incident_type', 'incident_state', 'incident_city', 'auto_make', 'policy_state', 'insured_occupation']:
            if col in df.columns:
                mask |= df[col].astype(str).str.lower().str.contains(search.lower(), na=False)
        df = df[mask]
    sort_col = request.args.get('sort', 'total_claim_amount')
    order = request.args.get('order', 'desc')
    if sort_col not in df.columns:
        sort_col = 'total_claim_amount'
    df = df.sort_values(by=sort_col, ascending=(order == 'asc'), na_position='last')  # type: ignore[call-overload]
    per_page = 10
    total = len(df)
    page = max(1, request.args.get('page', 1, type=int))
    start = (page - 1) * per_page
    subset = df.iloc[start:start + per_page]
    records = []
    for idx, row in subset.iterrows():
        r = row.to_dict()
        # Index can be any hashable; normalize to an int for templates
        idx_int = int(idx) if isinstance(idx, (int, np.integer)) else -1
        r['_idx'] = idx_int
        records.append(r)
    return render_template('claims.html', data=records, page=page, total_pages=max(1, (total + per_page - 1) // per_page), total=total, search=search, sort=sort_col, order=order, username=session.get('user'))


@app.route('/claim/<int:idx>')
@login_required
def claim_detail(idx):
    df = load_claims_data()
    if df is None or idx < 0 or idx >= len(df):
        return redirect(url_for('claims_list'))
    row = _normalize_fraud(df).iloc[idx].to_dict()
    row['_idx'] = idx
    return render_template('claim_detail.html', claim=row, username=session.get('user'))


@app.route('/dataset')
@login_required
def dataset():
    df = load_claims_data()
    if df is None:
        return render_template('dataset.html', data=[], columns=[], page=1, total_pages=0, total=0, username=session.get('user'))
    page = request.args.get('page', 1, type=int)
    per_page = 50
    total = len(df)
    start = (page - 1) * per_page
    subset = df.iloc[start:start + per_page]
    return render_template('dataset.html', data=subset.to_dict('records'), columns=subset.columns.tolist(), page=page, total_pages=max(1, (total + per_page - 1) // per_page), total=total, username=session.get('user'))


if __name__ == '__main__':
    init_db()
    # Eager-load model and data so the first page load isn't slow (avoids 15–30s wait)
    pkl_path = os.path.join(BASE_DIR, MODEL_FILE)
    if os.path.exists(pkl_path):
        try:
            print('Loading model and data (one-time, may take 10–20 seconds)...')
            get_models()
            load_claims_data()
            print('Ready. Starting server.\n')
        except Exception as e:
            print('Startup preload skipped:', e, '\n')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', port=port, debug=debug)
