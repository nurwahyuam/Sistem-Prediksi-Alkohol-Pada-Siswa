"""
preprocessing.py
Dataset sudah pre-normalized (0-1). 
Input baru dari user berupa raw values → encode → normalize → predict cluster.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import os

FEATURE_COLS = ['sex', 'address', 'Pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']

NORMALIZE_COLS = ['studytime', 'failures', 'famrel', 'freetime', 'goout', 'absences']
BINARY_COLS    = ['sex', 'address', 'Pstatus', 'romantic']

ENCODING_MAP = {
    'sex':      {'F': 0, 'M': 1},
    'address':  {'R': 0, 'U': 1},
    'Pstatus':  {'A': 0, 'T': 1},
    'romantic': {'No': 0, 'Yes': 1},
}

# Raw value ranges (sebelum normalisasi) dari dataset student-alcohol-consumption
RAW_RANGES = {
    'studytime': (1, 4),
    'failures':  (0, 3),
    'famrel':    (1, 5),
    'freetime':  (1, 5),
    'goout':     (1, 5),
    'absences':  (0, 93),
}

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'student_dataset.csv')
SAVED_MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def compute_label(dalc: float, walc: float):
    """
    score = 0.6*Dalc + 0.4*Walc  (range 1–5)
    Equal interval → 3 kategori
    """
    score  = 0.6 * dalc + 0.4 * walc
    lo, hi = 1.0, 5.0
    step   = (hi - lo) / 3
    t1, t2 = lo + step, lo + 2 * step   # ~2.333, ~3.667
    if score <= t1:
        cat = 'Normal'
    elif score <= t2:
        cat = 'Waspada'
    else:
        cat = 'Bahaya'
    return cat, round(score, 4), round(t1, 4), round(t2, 4)


def minmax_normalize_value(val: float, col: str) -> float:
    """Normalisasi satu nilai raw menggunakan rentang yang sudah diketahui."""
    lo, hi = RAW_RANGES[col]
    if hi == lo:
        return 0.0
    return round(float(val - lo) / float(hi - lo), 6)


def encode_new_case(raw_input: dict) -> dict:
    """
    Encode input kategorikal (string) → 0/1.
    Nilai numerik dibiarkan apa adanya (raw range).
    """
    encoded = {}
    for col in BINARY_COLS:
        val = raw_input[col]
        encoded[col] = ENCODING_MAP[col][val] if isinstance(val, str) else int(val)
    for col in NORMALIZE_COLS:
        encoded[col] = float(raw_input[col])
    return encoded


def normalize_new_case(encoded: dict) -> np.ndarray:
    """
    Normalisasi nilai numerik menggunakan RAW_RANGES.
    Binary cols (0/1) tidak perlu dinormalisasi lagi.
    """
    row = []
    for col in FEATURE_COLS:
        if col in BINARY_COLS:
            row.append(float(encoded[col]))
        else:
            row.append(minmax_normalize_value(encoded[col], col))
    return np.array(row, dtype=float)


def fit_and_save_scaler(df: pd.DataFrame) -> MinMaxScaler:
    """
    Dataset sudah normalized → scaler identity (min=0, max=1).
    Disimpan agar load_models() tetap bisa berjalan.
    """
    X = df[FEATURE_COLS].values
    scaler = MinMaxScaler()
    scaler.fit(X)
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(SAVED_MODEL_DIR, 'scaler.pkl'))
    return scaler


def compute_elbow(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    sse_list = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sse_list.append((k, round(km.inertia_, 4)))
    return sse_list


def find_optimal_k(sse_list: list, default_k: int = 3) -> int:
    if len(sse_list) < 3:
        return default_k
    sses = [s for _, s in sse_list]
    d1 = [sses[i] - sses[i+1] for i in range(len(sses)-1)]
    d2 = [d1[i] - d1[i+1] for i in range(len(d1)-1)]
    if not d2:
        return default_k
    elbow_idx = int(np.argmax(d2))
    k = elbow_idx + 2
    return k if 2 <= k <= 9 else default_k


def train_kmeans(df: pd.DataFrame, k: int) -> KMeans:
    X = df[FEATURE_COLS].values
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    joblib.dump(km, os.path.join(SAVED_MODEL_DIR, 'kmeans.pkl'))
    return km


def add_cluster_to_dataset(df: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    X = df[FEATURE_COLS].values
    df = df.copy()
    df['cluster'] = km.predict(X)
    return df


def load_models():
    sp = os.path.join(SAVED_MODEL_DIR, 'scaler.pkl')
    kp = os.path.join(SAVED_MODEL_DIR, 'kmeans.pkl')
    scaler = joblib.load(sp) if os.path.exists(sp) else None
    kmeans = joblib.load(kp) if os.path.exists(kp) else None
    return scaler, kmeans


def get_normalization_info(df: pd.DataFrame, scaler):
    """Tampilkan info normalisasi dari RAW_RANGES."""
    info = {}
    for col in FEATURE_COLS:
        if col in RAW_RANGES:
            lo, hi = RAW_RANGES[col]
            info[col] = {'min': lo, 'max': hi}
        else:
            info[col] = {'min': 0, 'max': 1}

    # Sample: ambil 5 baris dan tampilkan nilai normalized
    sample_records = []
    for _, row in df[FEATURE_COLS].head(5).iterrows():
        enc = row.to_dict()
        norm = normalize_new_case(enc)
        sample_records.append({col: round(float(norm[i]), 4)
                                for i, col in enumerate(FEATURE_COLS)})
    return info, sample_records
