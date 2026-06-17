"""
evaluation.py
=============
Modul Evaluasi Sistem CBR.

Alur:
1. Ambil seluruh case_base.xlsx
2. Split data 80:20 (training=case base, testing=evaluasi), stratify per label
3. Untuk setiap data testing, jalankan ulang logika REUSE
   (similarity tertinggi terhadap data training dalam cluster yang sama,
   TANPA voting / KNN) untuk memprediksi label
4. Bandingkan prediksi vs label asli
5. Hitung Accuracy, Precision, Recall, F1 Score, Confusion Matrix
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
CASE_BASE_XLSX = os.path.join(DATASET_DIR, 'case_base.xlsx')

FEATURE_COLS = ['sex', 'address', 'pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']
LABEL_ORDER = ['Normal', 'Waspada', 'Bahaya']


def _prediksi_reuse(fitur_vektor, cluster_id, df_train):
    """
    Replikasi logika Reuse: ambil label dari kasus training dengan
    cosine similarity TERTINGGI (tanpa voting / KNN).
    """
    df_cluster = df_train[df_train['cluster_id'] == cluster_id]
    if len(df_cluster) == 0:
        df_cluster = df_train

    matriks = df_cluster[FEATURE_COLS].values.astype(float)
    sims = cosine_similarity(fitur_vektor.reshape(1, -1), matriks)[0]

    idx_terbaik = int(np.argmax(sims))
    return df_cluster.iloc[idx_terbaik]['label']

def _simpan_eval_split(df_full, train_index, test_index):
    """
    Tandai hasil split evaluasi terakhir ke kolom 'eval_split' pada
    case_base.xlsx (TRAINING / TESTING), supaya terlihat di halaman
    Case Base kasus mana saja yang dipakai sebagai data testing.
    """
    df_full = df_full.copy()
    df_full['eval_split'] = 'TRAINING'
    df_full.loc[test_index, 'eval_split'] = 'TESTING'
    df_full.to_excel(CASE_BASE_XLSX, index=False)

def run_evaluation(test_size=0.2, random_state=42):
    """
    Jalankan evaluasi sistem dengan split 80:20.

    Returns:
        dict berisi total data, metrik evaluasi, dan confusion matrix.
    """
    if not os.path.exists(CASE_BASE_XLSX):
        raise ValueError('Case base belum tersedia.')

    df = pd.read_excel(CASE_BASE_XLSX)

    if len(df) < 10:
        raise ValueError('Jumlah data case base terlalu sedikit untuk evaluasi (minimal 10 data).')

    # Split 80:20 — stratify agar proporsi label terjaga
    try:
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['label']
        )
    except ValueError:
        # Fallback jika ada label dengan jumlah terlalu sedikit untuk stratify
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    y_true, y_pred = [], []

    for _, row in df_test.iterrows():
        fitur_vektor = row[FEATURE_COLS].values.astype(float)
        label_prediksi = _prediksi_reuse(fitur_vektor, row['cluster_id'], df_train)
        y_true.append(row['label'])
        y_pred.append(label_prediksi)

    # Tandai status TRAINING/TESTING hasil split ini ke case_base.xlsx
    _simpan_eval_split(df, df_train.index, df_test.index)

    labels_ada = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels_ada, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels_ada, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels_ada, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_ada)

    return {
        'total_data': len(df),
        'total_train': len(df_train),
        'total_test': len(df_test),
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'labels': labels_ada,
        'confusion_matrix': cm.tolist(),
        'detail': [
            {'actual': a, 'predicted': p, 'benar': a == p}
            for a, p in zip(y_true, y_pred)
        ],
    }
