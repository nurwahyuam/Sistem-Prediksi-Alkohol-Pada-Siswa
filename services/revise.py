"""
revise.py
=========
Modul CBR - Tahap REVISE

Alur:
1. Pakar melihat prediksi sistem + detail kasus
2. Pakar bisa mengubah label atau mempertahankan prediksi sistem
3. Pakar bisa menulis catatan
4. Simpan revisi ke revision_log.xlsx
"""

import pandas as pd
import os
from datetime import datetime

# ===== PATH KONFIGURASI =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
REVISION_LOG_XLSX = os.path.join(DATASET_DIR, 'revision_log.xlsx')

LABEL_OPTIONS = ['Normal', 'Waspada', 'Bahaya']


def simpan_revisi(case_id, system_label, expert_label, note):
    """
    Simpan hasil revisi pakar ke revision_log.xlsx.
    
    Parameters:
        case_id: ID kasus yang direvisi
        system_label: label dari prediksi sistem
        expert_label: label yang dipilih pakar
        note: catatan dari pakar
    
    Returns:
        dict berisi revision_id dan status
    """
    # Load atau buat file revision_log
    if os.path.exists(REVISION_LOG_XLSX):
        df_log = pd.read_excel(REVISION_LOG_XLSX)
        # Generate revision_id berikutnya
        n = len(df_log) + 1
    else:
        df_log = pd.DataFrame(columns=[
            'revision_id', 'case_id', 'system_label',
            'expert_label', 'note', 'created_at'
        ])
        n = 1

    revision_id = f'REV-{str(n).zfill(3)}'

    # Record baru
    new_record = {
        'revision_id': revision_id,
        'case_id': case_id,
        'system_label': system_label,
        'expert_label': expert_label,
        'note': note if note else '-',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    df_log = pd.concat([df_log, pd.DataFrame([new_record])], ignore_index=True)
    df_log.to_excel(REVISION_LOG_XLSX, index=False)

    return {
        'revision_id': revision_id,
        'label_berubah': system_label != expert_label,
        'status_kasus': 'EXPERT_REVISED' if system_label != expert_label else 'VALIDATED'
    }


def load_revision_log():
    """Load revision log dari Excel."""
    if not os.path.exists(REVISION_LOG_XLSX):
        return pd.DataFrame()
    return pd.read_excel(REVISION_LOG_XLSX)


def get_label_options():
    """Kembalikan pilihan label yang tersedia."""
    return LABEL_OPTIONS
