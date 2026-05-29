"""
cbr_engine.py
Implementasi penuh Case Based Reasoning:
- Retrieve: ambil kasus dengan cluster sama
- Reuse: cosine similarity
- Revise: evaluasi similarity tertinggi
- Retain: simpan kasus baru ke CSV
"""

import numpy as np
import pandas as pd
import os

from utils.preprocessing import FEATURE_COLS, DATASET_PATH

LOW_SIMILARITY_THRESHOLD = 0.5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hitung cosine similarity antara dua vektor.
    similarity = (A·B) / (||A|| × ||B||)
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve(df_with_cluster: pd.DataFrame, new_cluster: int) -> pd.DataFrame:
    """
    RETRIEVE: Ambil semua kasus dari cluster yang sama.
    """
    same_cluster = df_with_cluster[df_with_cluster['cluster'] == new_cluster].copy()
    return same_cluster.reset_index(drop=True)


def reuse(retrieved_df: pd.DataFrame, new_case_normalized: np.ndarray, top_n: int = 5):
    """
    REUSE: Hitung cosine similarity antara kasus baru dengan semua kasus dalam cluster.
    Return list top_n kasus paling mirip beserta similarity score.
    """
    results = []
    feature_vals = retrieved_df[FEATURE_COLS].values

    for idx, row_vals in enumerate(feature_vals):
        sim = cosine_similarity(new_case_normalized, row_vals)
        results.append({
            'index': idx,
            'similarity': round(sim, 6),
            'label_sentimen': retrieved_df.iloc[idx]['label_sentimen'],
            **{col: round(float(retrieved_df.iloc[idx][col]), 4) for col in FEATURE_COLS}
        })

    # Urutkan similarity tertinggi
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def revise(top_cases: list) -> dict:
    """
    REVISE: Evaluasi kasus dengan similarity tertinggi.
    Jika similarity < threshold, beri warning.
    Return: kategori prediksi dan warning flag.
    """
    if not top_cases:
        return {'kategori': 'Unknown', 'warning': True, 'top_case': None}

    best = top_cases[0]
    warning = best['similarity'] < LOW_SIMILARITY_THRESHOLD

    return {
        'kategori': best['label_sentimen'],
        'warning': warning,
        'top_case': best,
        'similarity_score': best['similarity'],
    }


def retain(new_case_encoded: dict, kategori: str, cluster: int):
    """
    RETAIN: Simpan kasus baru ke dataset CSV.
    new_case_encoded sudah dinormalisasi.
    """
    new_row = {col: new_case_encoded.get(col, 0) for col in FEATURE_COLS}
    new_row['label_sentimen'] = kategori
    new_row['cluster'] = cluster

    df = pd.read_csv(DATASET_PATH)

    # Tambahkan cluster jika belum ada
    if 'cluster' not in df.columns:
        df['cluster'] = -1

    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATASET_PATH, index=False)
    return True


def run_cbr(df_with_cluster: pd.DataFrame,
            new_case_normalized: np.ndarray,
            new_cluster: int,
            new_case_normalized_dict: dict,
            kategori_prediksi: str) -> dict:
    """
    Jalankan full pipeline CBR dan return semua hasil.
    """
    # RETRIEVE
    retrieved = retrieve(df_with_cluster, new_cluster)

    # REUSE
    top_cases = reuse(retrieved, new_case_normalized, top_n=5)

    # REVISE
    revision = revise(top_cases)

    return {
        'retrieve': {
            'cluster': new_cluster,
            'total_retrieved': len(retrieved),
        },
        'reuse': {
            'top_cases': top_cases,
        },
        'revise': {
            'kategori': revision['kategori'],
            'warning': revision['warning'],
            'top_case': revision['top_case'],
            'similarity_score': revision.get('similarity_score', 0),
        },
        'retain_ready': True,
    }
