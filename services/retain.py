"""
retain.py
=========
Modul CBR - Tahap RETAIN

Alur:
1. Setelah pakar menekan tombol Simpan:
   - Jika label berubah → status = EXPERT_REVISED
   - Jika label sama   → status = VALIDATED
2. Tambahkan kasus baru ke case_base.xlsx
3. Tambahkan ke retrieve_history.xlsx
4. Knowledge base bertambah!
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
CASE_BASE_XLSX = os.path.join(DATASET_DIR, 'case_base.xlsx')
RETRIEVE_HISTORY_XLSX = os.path.join(DATASET_DIR, 'retrieve_history.xlsx')

FEATURE_COLS = ['sex', 'address', 'pstatus', 'studytime', 'failures',
                'romantic', 'famrel', 'freetime', 'goout', 'absences']


def generate_case_id():
    """
    Generate case_id baru berdasarkan jumlah data di case_base.
    Format: CB-001, CB-002, ...
    """
    if not os.path.exists(CASE_BASE_XLSX):
        return 'CB-001'

    df = pd.read_excel(CASE_BASE_XLSX)
    n = len(df) + 1
    return f'CB-{str(n).zfill(3)}'


def generate_history_id():
    """
    Generate history_id baru.
    Format: HIS-001, HIS-002, ...
    """
    if not os.path.exists(RETRIEVE_HISTORY_XLSX):
        return 'HIS-001'

    df = pd.read_excel(RETRIEVE_HISTORY_XLSX)
    n = len(df) + 1
    return f'HIS-{str(n).zfill(3)}'


def run_retain(fitur_normalized_dict, cluster_id, alcohol_score,
               system_label, final_label, top_cases):
    """
    Jalankan proses Retain CBR.
    
    Parameters:
        fitur_normalized_dict: dict fitur kasus baru (sudah dinormalisasi)
        cluster_id: cluster yang diprediksi
        alcohol_score: skor alkohol kasus baru
        system_label: label dari prediksi sistem
        final_label: label final (dari pakar atau sistem)
        top_cases: list top kasus hasil retrieve
    
    Returns:
        dict berisi case_id baru dan info retain
    """
    # 1. Tentukan status
    label_berubah = system_label != final_label
    status = 'EXPERT_REVISED' if label_berubah else 'VALIDATED'

    # 2. Generate case_id baru
    case_id = generate_case_id()

    # 3. Tambahkan ke case_base.xlsx
    df_case = pd.read_excel(CASE_BASE_XLSX)

    new_case = {
        'case_id': case_id,
        'cluster_id': cluster_id,
        'sex': fitur_normalized_dict['sex'],
        'address': fitur_normalized_dict['address'],
        'pstatus': fitur_normalized_dict['pstatus'],
        'studytime': fitur_normalized_dict['studytime'],
        'failures': fitur_normalized_dict['failures'],
        'romantic': fitur_normalized_dict['romantic'],
        'famrel': fitur_normalized_dict['famrel'],
        'freetime': fitur_normalized_dict['freetime'],
        'goout': fitur_normalized_dict['goout'],
        'absences': fitur_normalized_dict['absences'],
        'alcohol_score': alcohol_score,
        'label': final_label,
        'status': status,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    df_case = pd.concat([df_case, pd.DataFrame([new_case])], ignore_index=True)
    df_case.to_excel(CASE_BASE_XLSX, index=False)

    # 4. Tambahkan ke retrieve_history.xlsx
    history_id = generate_history_id()

    top1 = top_cases[0] if top_cases else {}

    history_record = {
        'history_id': history_id,
        'case_id': case_id,
        'top1_case_id': top1.get('case_id', '-'),
        'top1_similarity': top1.get('similarity', 0),
        'cluster_id': cluster_id,
        'system_label': system_label,
        'final_label': final_label,
        'status': status,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if os.path.exists(RETRIEVE_HISTORY_XLSX):
        df_history = pd.read_excel(RETRIEVE_HISTORY_XLSX)
    else:
        df_history = pd.DataFrame(columns=[
            'history_id', 'case_id', 'top1_case_id', 'top1_similarity',
            'cluster_id', 'system_label', 'final_label', 'status', 'created_at'
        ])

    df_history = pd.concat([df_history, pd.DataFrame([history_record])], ignore_index=True)
    df_history.to_excel(RETRIEVE_HISTORY_XLSX, index=False)

    return {
        'case_id': case_id,
        'history_id': history_id,
        'status': status,
        'label_berubah': label_berubah,
        'total_case_base': len(df_case),
        'pesan': (
            f"Kasus {case_id} berhasil disimpan ke Case Base dengan status {status}. "
            f"Knowledge base sekarang memiliki {len(df_case)} kasus."
        )
    }


def load_retrieve_history():
    """Load retrieve history dari Excel."""
    if not os.path.exists(RETRIEVE_HISTORY_XLSX):
        return pd.DataFrame()
    return pd.read_excel(RETRIEVE_HISTORY_XLSX)
