"""
expert.py
Manajemen pakar: autentikasi, CRUD basis kasus, riwayat CBR, revisi label.
"""

import json
import os
import hashlib
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS_PATH = os.path.join(BASE_DIR, 'dataset', 'experts.json')
HISTORY_PATH = os.path.join(BASE_DIR, 'dataset', 'cbr_history.json')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: str, data) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_expert(username: str, password: str):
    """
    Verifikasi username dan password pakar.
    Return: dict pakar jika cocok, None jika tidak.
    """
    experts = _load_json(EXPERTS_PATH)
    for e in experts:
        if e['username'] == username and e['password'] == password:
            return e
    return None


def get_all_experts() -> list:
    return _load_json(EXPERTS_PATH)


def add_expert(username: str, password: str, nama: str, jabatan: str) -> tuple[bool, str]:
    experts = _load_json(EXPERTS_PATH)
    if any(e['username'] == username for e in experts):
        return False, f"Username '{username}' sudah digunakan."
    experts.append({
        'username': username,
        'password': password,
        'nama': nama,
        'jabatan': jabatan,
        'created_at': _now()[:10],
    })
    _save_json(EXPERTS_PATH, experts)
    return True, 'Pakar berhasil ditambahkan.'


def delete_expert(username: str) -> tuple[bool, str]:
    if username == 'admin':
        return False, 'Akun admin tidak bisa dihapus.'
    experts = _load_json(EXPERTS_PATH)
    new = [e for e in experts if e['username'] != username]
    if len(new) == len(experts):
        return False, 'Username tidak ditemukan.'
    _save_json(EXPERTS_PATH, new)
    return True, 'Pakar berhasil dihapus.'


def update_expert_password(username: str, new_password: str) -> tuple[bool, str]:
    experts = _load_json(EXPERTS_PATH)
    for e in experts:
        if e['username'] == username:
            e['password'] = new_password
            _save_json(EXPERTS_PATH, experts)
            return True, 'Password berhasil diubah.'
    return False, 'Username tidak ditemukan.'


# ── CBR History ───────────────────────────────────────────────────────────────

def save_cbr_history(entry: dict) -> None:
    """Simpan satu record riwayat CBR."""
    history = _load_json(HISTORY_PATH)
    entry['timestamp'] = _now()
    entry['id'] = len(history) + 1
    history.append(entry)
    _save_json(HISTORY_PATH, history)


def get_cbr_history(limit: int = 50) -> list:
    history = _load_json(HISTORY_PATH)
    return list(reversed(history))[:limit]


def clear_cbr_history() -> None:
    _save_json(HISTORY_PATH, [])


# ── Basis Kasus Management ────────────────────────────────────────────────────

def revise_label(dataset_path: str, row_index: int, new_label: str,
                 pakar_username: str) -> tuple[bool, str]:
    """
    Pakar merevisi label_sentimen pada baris tertentu di dataset CSV.
    """
    import pandas as pd
    if new_label not in ('Normal', 'Waspada', 'Bahaya'):
        return False, 'Label tidak valid. Pilih: Normal, Waspada, atau Bahaya.'
    try:
        df = pd.read_csv(dataset_path)
        if row_index < 0 or row_index >= len(df):
            return False, f'Index baris {row_index} tidak valid.'
        old_label = df.at[row_index, 'label_sentimen']
        df.at[row_index, 'label_sentimen'] = new_label
        df.to_csv(dataset_path, index=False)
        # Catat ke history
        save_cbr_history({
            'type': 'revisi_label',
            'pakar': pakar_username,
            'row_index': row_index,
            'label_lama': old_label,
            'label_baru': new_label,
        })
        return True, f'Label baris {row_index} diubah dari {old_label} → {new_label}.'
    except Exception as ex:
        return False, str(ex)


def delete_case(dataset_path: str, row_index: int,
                pakar_username: str) -> tuple[bool, str]:
    """Hapus satu baris kasus dari dataset."""
    import pandas as pd
    try:
        df = pd.read_csv(dataset_path)
        if row_index < 0 or row_index >= len(df):
            return False, 'Index tidak valid.'
        removed = df.iloc[row_index].to_dict()
        df = df.drop(index=row_index).reset_index(drop=True)
        df.to_csv(dataset_path, index=False)
        save_cbr_history({
            'type': 'hapus_kasus',
            'pakar': pakar_username,
            'row_index': row_index,
            'data': str(removed),
        })
        return True, f'Baris {row_index} berhasil dihapus.'
    except Exception as ex:
        return False, str(ex)


def add_manual_case(dataset_path: str, row_data: dict,
                    pakar_username: str) -> tuple[bool, str]:
    """Pakar menambahkan kasus manual ke dataset."""
    import pandas as pd
    try:
        df = pd.read_csv(dataset_path)
        new_row = pd.DataFrame([row_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(dataset_path, index=False)
        save_cbr_history({
            'type': 'tambah_kasus_manual',
            'pakar': pakar_username,
            'data': str(row_data),
        })
        return True, 'Kasus manual berhasil ditambahkan.'
    except Exception as ex:
        return False, str(ex)


def get_label_stats(dataset_path: str) -> dict:
    """Statistik distribusi label di dataset."""
    import pandas as pd
    df = pd.read_csv(dataset_path)
    counts = df['label_sentimen'].value_counts().to_dict()
    total = len(df)
    return {
        'total': total,
        'counts': counts,
        'pct': {k: round(v / total * 100, 1) for k, v in counts.items()},
    }
