"""
clustering.py
=============
Modul untuk K-Means Clustering dengan Elbow Method.

Tahapan:
1. Load case_base.xlsx
2. Jalankan Elbow Method (K=1 sampai 10) → hitung SSE
3. Tentukan K optimal (titik elbow)
4. Train KMeans dengan K optimal
5. Simpan model ke models/kmeans.pkl
6. Update cluster_id di case_base.xlsx
7. Simpan info cluster ke cluster_info.xlsx
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.cluster import KMeans

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CASE_BASE_XLSX = os.path.join(DATASET_DIR, 'case_base.xlsx')
CLUSTER_INFO_XLSX = os.path.join(DATASET_DIR, 'cluster_info.xlsx')
KMEANS_PKL = os.path.join(MODEL_DIR, 'kmeans.pkl')

# Kolom fitur yang dipakai untuk clustering
FEATURE_COLS = ['sex', 'address', 'pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']


def hitung_elbow(df_features, k_min=1, k_max=10):
    """
    Hitung SSE untuk setiap nilai K dari k_min sampai k_max.
    
    Parameters:
        df_features: DataFrame berisi fitur yang sudah dinormalisasi
        k_min: nilai K minimum
        k_max: nilai K maksimum
    
    Returns:
        dict berisi list K dan list SSE
    """
    k_values = list(range(k_min, k_max + 1))
    sse_values = []

    X = df_features[FEATURE_COLS].values

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse_values.append(round(kmeans.inertia_, 4))

    return {
        'k_values': k_values,
        'sse_values': sse_values
    }


def tentukan_k_optimal(k_values, sse_values):
    """
    Menentukan titik elbow menggunakan perubahan penurunan SSE.
    """

    if len(sse_values) < 3:
        return 3

    deltas = []

    for i in range(1, len(sse_values)):
        deltas.append(sse_values[i-1] - sse_values[i])

    elbow_scores = []

    for i in range(1, len(deltas)):
        if deltas[i-1] == 0:
            elbow_scores.append(0)
        else:
            score = deltas[i] / deltas[i-1]
            elbow_scores.append(score)

    idx = elbow_scores.index(min(elbow_scores))

    return k_values[idx + 2]


def train_kmeans(k_optimal, df_features):
    """
    Train KMeans dengan K optimal.
    
    Parameters:
        k_optimal: jumlah cluster
        df_features: DataFrame dengan kolom fitur
    
    Returns:
        model KMeans yang sudah dilatih
    """
    X = df_features[FEATURE_COLS].values
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


def simpan_model(kmeans_model):
    """Simpan model KMeans ke file pkl."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(KMEANS_PKL, 'wb') as f:
        pickle.dump(kmeans_model, f)


def load_model():
    """Load model KMeans dari file pkl."""
    if not os.path.exists(KMEANS_PKL):
        return None
    with open(KMEANS_PKL, 'rb') as f:
        return pickle.load(f)


def update_cluster_case_base(kmeans_model):
    """
    Update kolom cluster_id di case_base.xlsx menggunakan model yang sudah dilatih.
    """
    df = pd.read_excel(CASE_BASE_XLSX)
    X = df[FEATURE_COLS].values
    df['cluster_id'] = kmeans_model.predict(X)
    df.to_excel(CASE_BASE_XLSX, index=False)
    return df


def simpan_cluster_info(kmeans_model, k_optimal, df_case_base):
    """
    Simpan informasi cluster ke cluster_info.xlsx.
    """
    records = []
    for cluster_id in range(k_optimal):
        df_cluster = df_case_base[df_case_base['cluster_id'] == cluster_id]
        total = len(df_cluster)

        # Label dominan di cluster ini
        if total > 0:
            dominant = df_cluster['label'].mode()[0]
        else:
            dominant = '-'

        records.append({
            'cluster_id': cluster_id,
            'k_optimal': k_optimal,
            'total_data': total,
            'dominant_label': dominant,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    df_info = pd.DataFrame(records)
    df_info.to_excel(CLUSTER_INFO_XLSX, index=False)
    return df_info


def run_clustering():
    """
    Fungsi utama: jalankan seluruh pipeline clustering.
    
    Returns:
        dict berisi hasil elbow, K optimal, distribusi cluster
    """
    # 1. Load case base
    df = pd.read_excel(CASE_BASE_XLSX)

    # 2. Hitung Elbow
    elbow_data = hitung_elbow(df, k_min=1, k_max=10)
    k_values = elbow_data['k_values']
    sse_values = elbow_data['sse_values']

    # 3. Tentukan K optimal
    k_optimal = tentukan_k_optimal(k_values, sse_values)

    # 4. Train KMeans
    kmeans_model = train_kmeans(k_optimal, df)

    # 5. Simpan model
    simpan_model(kmeans_model)

    # 6. Update cluster_id di case_base
    df_updated = update_cluster_case_base(kmeans_model)

    # 7. Simpan cluster info
    df_info = simpan_cluster_info(kmeans_model, k_optimal, df_updated)

    # 8. Hitung distribusi per cluster
    distribusi = df_updated.groupby('cluster_id')['label'].value_counts().to_dict()
    distribusi_str = {str(k): str(v) for k, v in distribusi.items()}

    # Hitung persen penurunan SSE untuk ditampilkan di grafik
    penurunan_pct = []
    for i in range(1, len(sse_values)):
        if sse_values[i-1] > 0:
            pct = ((sse_values[i-1] - sse_values[i]) / sse_values[i-1]) * 100
            penurunan_pct.append(round(pct, 2))
        else:
            penurunan_pct.append(0)

    return {
        'k_optimal': k_optimal,
        'k_values': k_values,
        'sse_values': sse_values,
        'penurunan_pct': penurunan_pct,
        'cluster_info': df_info.to_dict('records'),
        'total_data': len(df),
    }


def prediksi_cluster_baru(fitur_normalized_dict):
    """
    Prediksi cluster untuk kasus baru.
    
    Parameters:
        fitur_normalized_dict: dict fitur yang sudah dinormalisasi
    
    Returns:
        int: cluster_id prediksi
    """
    model = load_model()
    if model is None:
        raise ValueError("Model belum dilatih. Jalankan Training terlebih dahulu.")

    X = np.array([[fitur_normalized_dict[col] for col in FEATURE_COLS]])
    cluster = model.predict(X)[0]
    return int(cluster)


def cek_training_selesai():
    """Cek apakah training sudah dilakukan."""
    return os.path.exists(KMEANS_PKL) and os.path.exists(CLUSTER_INFO_XLSX)


def load_cluster_info():
    """Load cluster info dari Excel."""
    if not os.path.exists(CLUSTER_INFO_XLSX):
        return None
    return pd.read_excel(CLUSTER_INFO_XLSX)
