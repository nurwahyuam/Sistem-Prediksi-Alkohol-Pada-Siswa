"""
preprocessing.py
================
Modul untuk preprocessing dataset student-mat.csv

Tahapan:
1. Encoding kolom kategorikal
2. Hitung alcohol_score = 0.6*Dalc + 0.4*Walc
3. Pelabelan dengan Equal Interval (Normal/Waspada/Bahaya)
4. Normalisasi Min-Max untuk kolom numerik
5. Simpan hasil ke dataset_preprocessed.xlsx
6. Buat case_base.xlsx dari hasil preprocessing
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
RAW_CSV = os.path.join(DATASET_DIR, 'student-mat.csv')
PREPROCESSED_XLSX = os.path.join(DATASET_DIR, 'dataset_preprocessed.xlsx')
CASE_BASE_XLSX = os.path.join(DATASET_DIR, 'case_base.xlsx')

# ===== NILAI MIN-MAX TETAP (sesuai PPT kelompok) =====
MIN_MAX_VALUES = {
    'studytime': {'min': 1, 'max': 4},
    'failures':  {'min': 0, 'max': 3},
    'famrel':    {'min': 1, 'max': 5},
    'freetime':  {'min': 1, 'max': 5},
    'goout':     {'min': 1, 'max': 5},
    'absences':  {'min': 0, 'max': 75},
}

# ===== KOLOM FITUR YANG DIGUNAKAN =====
FEATURE_COLS = ['sex', 'address', 'pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']


def encode_kategorical(df):
    """
    Encoding kolom kategorikal menjadi numerik.
    
    sex:      F=0, M=1
    address:  R=0, U=1
    Pstatus:  A=0, T=1
    romantic: no=0, yes=1
    """
    df = df.copy()

    # Rename kolom Pstatus -> pstatus agar konsisten lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Encoding
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    df['address'] = df['address'].map({'R': 0, 'U': 1})
    df['pstatus'] = df['pstatus'].map({'A': 0, 'T': 1})
    df['romantic'] = df['romantic'].map({'no': 0, 'yes': 1})

    return df


def hitung_alcohol_score(df):
    """
    Hitung alcohol_score = 0.6 * Dalc + 0.4 * Walc
    """
    df = df.copy()
    df['alcohol_score'] = (0.6 * df['dalc']) + (0.4 * df['walc'])
    return df


def labeling_equal_interval(df):
    """
    Pelabelan kategori menggunakan Equal Interval.
    
    Dari PPT:
    - alcohol_score range: 1.0 - 5.0
    - Interval = (5 - 1) / 3 = 1.33
    - Normal:  1.00 - 2.33
    - Waspada: 2.34 - 3.67
    - Bahaya:  3.68 - 5.00
    """
    df = df.copy()

    x_min = 1.0
    x_max = 5.0
    k = 3
    interval = (x_max - x_min) / k  # = 1.333...

    batas1 = x_min + interval        # 2.333
    batas2 = x_min + 2 * interval    # 3.667

    def tentukan_label(score):
        if score <= batas1:
            return 'Normal'
        elif score <= batas2:
            return 'Waspada'
        else:
            return 'Bahaya'

    df['label'] = df['alcohol_score'].apply(tentukan_label)
    return df, interval, batas1, batas2


def normalisasi_minmax(df):
    """
    Normalisasi Min-Max untuk kolom numerik.
    
    Rumus: x' = (x - min) / (max - min)
    
    Menggunakan nilai min-max tetap sesuai PPT.
    """
    df = df.copy()

    for col, bounds in MIN_MAX_VALUES.items():
        mn = bounds['min']
        mx = bounds['max']
        if mx != mn:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 0.0

    # Bulatkan ke 4 desimal
    for col in MIN_MAX_VALUES.keys():
        df[col] = df[col].round(4)

    return df


def normalisasi_satu_kasus(data_dict):
    """
    Normalisasi satu kasus baru (dict input dari user).
    
    Parameters:
        data_dict: dict dengan key = nama fitur (raw value)
    
    Returns:
        dict dengan nilai ternormalisasi
    """
    result = {}

    # Encoding
    result['sex'] = 0 if data_dict.get('sex', 'F') == 'F' else 1
    result['address'] = 0 if data_dict.get('address', 'R') == 'R' else 1
    result['pstatus'] = 0 if data_dict.get('pstatus', 'A') == 'A' else 1
    result['romantic'] = 0 if data_dict.get('romantic', 'no').lower() == 'no' else 1

    # Normalisasi numerik
    numerik = ['studytime', 'failures', 'famrel', 'freetime', 'goout', 'absences']
    for col in numerik:
        val = float(data_dict.get(col, 0))
        mn = MIN_MAX_VALUES[col]['min']
        mx = MIN_MAX_VALUES[col]['max']
        if mx != mn:
            result[col] = round((val - mn) / (mx - mn), 4)
        else:
            result[col] = 0.0

    return result


def run_preprocessing():
    """
    Fungsi utama: jalankan seluruh pipeline preprocessing.
    
    Returns:
        dict berisi info hasil preprocessing
    """
    # 1. Baca CSV
    df_raw = pd.read_csv(RAW_CSV, sep=';')
    total_raw = len(df_raw)

    # 2. Pilih kolom yang diperlukan
    kolom_pakai = ['sex', 'address', 'Pstatus', 'studytime', 'failures',
                   'romantic', 'famrel', 'freetime', 'goout', 'absences', 'Dalc', 'Walc']
    df = df_raw[kolom_pakai].copy()

    # 3. Encoding
    df = encode_kategorical(df)

    # 4. Hitung alcohol_score
    df = hitung_alcohol_score(df)

    # 5. Pelabelan
    df, interval, batas1, batas2 = labeling_equal_interval(df)

    # 6. Normalisasi
    df = normalisasi_minmax(df)

    # 7. Hapus kolom dalc, walc (sudah diwakili alcohol_score)
    df = df.drop(columns=['dalc', 'walc'])

    # 8. Simpan dataset_preprocessed.xlsx
    df.to_excel(PREPROCESSED_XLSX, index=False)

    # 9. Buat case_base.xlsx
    buat_case_base(df)

    # Hitung distribusi label
    distribusi = df['label'].value_counts().to_dict()

    return {
        'total_data': total_raw,
        'distribusi_label': distribusi,
        'interval': round(interval, 4),
        'batas_normal': f'1.00 - {batas1:.2f}',
        'batas_waspada': f'{batas1+0.01:.2f} - {batas2:.2f}',
        'batas_bahaya': f'{batas2+0.01:.2f} - 5.00',
        'file_output': PREPROCESSED_XLSX,
        'case_base_output': CASE_BASE_XLSX,
    }


def buat_case_base(df_preprocessed):
    """
    Buat file case_base.xlsx dari dataset_preprocessed.
    
    case_base berbeda dari dataset_preprocessed karena:
    - Ada kolom case_id (CB-001, CB-002, ...)
    - Ada kolom cluster_id (diisi setelah training)
    - Ada kolom status (TRAINING)
    - Ada kolom created_at
    """
    df = df_preprocessed.copy()
    n = len(df)

    # Buat case_id
    df.insert(0, 'case_id', [f'CB-{str(i+1).zfill(3)}' for i in range(n)])
    df.insert(1, 'cluster_id', -1)  # -1 berarti belum di-cluster

    # Pastikan urutan kolom sesuai spesifikasi
    kolom_order = [
        'case_id', 'cluster_id',
        'sex', 'address', 'pstatus', 'studytime', 'failures',
        'romantic', 'famrel', 'freetime', 'goout', 'absences',
        'alcohol_score', 'label', 'status', 'created_at'
    ]

    df['status'] = 'TRAINING'
    df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    df = df[kolom_order]
    df.to_excel(CASE_BASE_XLSX, index=False)


def load_preprocessed():
    """Load dataset preprocessed dari Excel."""
    if not os.path.exists(PREPROCESSED_XLSX):
        return None
    return pd.read_excel(PREPROCESSED_XLSX)


def load_case_base():
    """Load case base dari Excel."""
    if not os.path.exists(CASE_BASE_XLSX):
        return None
    return pd.read_excel(CASE_BASE_XLSX)


def cek_preprocessing_selesai():
    """Cek apakah preprocessing sudah pernah dijalankan."""
    return os.path.exists(PREPROCESSED_XLSX) and os.path.exists(CASE_BASE_XLSX)
