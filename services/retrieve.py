"""
retrieve.py
===========
Modul CBR - Tahap RETRIEVE

Alur:
1. Terima input kasus baru (sudah dipreprocessing)
2. Prediksi cluster kasus baru menggunakan model KMeans
3. Ambil semua kasus dari case_base yang cluster-nya sama
4. Hitung Cosine Similarity antara kasus baru dan setiap kasus dalam cluster
5. Kembalikan Top 5 kasus dengan similarity tertinggi
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
CASE_BASE_XLSX = os.path.join(DATASET_DIR, 'case_base.xlsx')

# Kolom fitur yang dipakai untuk perhitungan similarity
FEATURE_COLS = ['sex', 'address', 'pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']


def hitung_cosine_similarity(vektor_baru, vektor_lama):
    """
    Hitung cosine similarity antara dua vektor.
    
    Rumus: similarity = (A · B) / (||A|| × ||B||)
    
    Parameters:
        vektor_baru: numpy array (1D) - vektor kasus baru
        vektor_lama: numpy array (2D) - matriks kasus lama
    
    Returns:
        numpy array berisi nilai similarity
    """
    # reshape untuk sklearn
    A = vektor_baru.reshape(1, -1)
    B = vektor_lama

    similarities = cosine_similarity(A, B)[0]
    return similarities


def run_retrieve(fitur_normalized_dict, cluster_id_baru, top_n=5):
    """
    Jalankan proses Retrieve CBR.
    
    Parameters:
        fitur_normalized_dict: dict fitur kasus baru yang sudah dinormalisasi
        cluster_id_baru: cluster yang diprediksi untuk kasus baru
        top_n: jumlah kasus terbaik yang dikembalikan (default 5)
    
    Returns:
        dict berisi:
        - top_cases: list dict kasus dengan similarity tertinggi
        - cluster_id: cluster kasus baru
        - total_dalam_cluster: jumlah kasus dalam cluster
    """
    # 1. Load case base
    df_case = pd.read_excel(CASE_BASE_XLSX)

    # 2. Filter kasus dalam cluster yang sama
    df_cluster = df_case[df_case['cluster_id'] == cluster_id_baru].copy()

    if len(df_cluster) == 0:
        # Jika cluster kosong, gunakan semua data
        df_cluster = df_case.copy()

    # 3. Buat vektor kasus baru
    vektor_baru = np.array([fitur_normalized_dict[col] for col in FEATURE_COLS])

    # 4. Buat matriks kasus lama
    matriks_lama = df_cluster[FEATURE_COLS].values

    # 5. Hitung cosine similarity
    similarities = hitung_cosine_similarity(vektor_baru, matriks_lama)

    # 6. Tambahkan kolom similarity ke dataframe
    df_cluster = df_cluster.copy()
    df_cluster['similarity'] = similarities
    df_cluster['similarity'] = df_cluster['similarity'].round(4)

    # 7. Sort descending berdasarkan similarity
    df_cluster = df_cluster.sort_values('similarity', ascending=False)

    # 8. Ambil Top N
    top_cases = df_cluster.head(top_n)

    # 9. Format hasil
    hasil = []
    for _, row in top_cases.iterrows():
        hasil.append({
            'case_id': row['case_id'],
            'cluster_id': int(row['cluster_id']),
            'similarity': float(row['similarity']),
            'similarity_pct': f"{float(row['similarity'])*100:.2f}%",
            'label': row['label'],
            'status': row['status'],
            # Fitur asli (untuk ditampilkan)
            'sex': int(row['sex']),
            'address': int(row['address']),
            'pstatus': int(row['pstatus']),
            'studytime': float(row['studytime']),
            'failures': float(row['failures']),
            'romantic': int(row['romantic']),
            'famrel': float(row['famrel']),
            'freetime': float(row['freetime']),
            'goout': float(row['goout']),
            'absences': float(row['absences']),
            'alcohol_score': float(row['alcohol_score']),
        })

    return {
        'top_cases': hasil,
        'cluster_id': cluster_id_baru,
        'total_dalam_cluster': len(df_cluster),
        'total_case_base': len(df_case),
    }


def decode_fitur(fitur_dict):
    """
    Decode fitur dari numerik ke label asli (untuk tampilan).
    Ini untuk menampilkan nilai asli (sebelum encoding) di UI.
    """
    decoded = {}
    decoded['sex'] = 'F' if fitur_dict.get('sex', 0) == 0 else 'M'
    decoded['address'] = 'Rural (R)' if fitur_dict.get('address', 0) == 0 else 'Urban (U)'
    decoded['pstatus'] = 'Apart (A)' if fitur_dict.get('pstatus', 0) == 0 else 'Together (T)'
    decoded['romantic'] = 'No' if fitur_dict.get('romantic', 0) == 0 else 'Yes'
    decoded['studytime'] = fitur_dict.get('studytime', 0)
    decoded['failures'] = fitur_dict.get('failures', 0)
    decoded['famrel'] = fitur_dict.get('famrel', 0)
    decoded['freetime'] = fitur_dict.get('freetime', 0)
    decoded['goout'] = fitur_dict.get('goout', 0)
    decoded['absences'] = fitur_dict.get('absences', 0)
    return decoded
