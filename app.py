"""
app.py — Main Flask Application
CBR Sistem Klasifikasi Risiko Alkohol Siswa
+ Ruang Pakar dengan proses REVISE interaktif
"""

import os
import json
import numpy as np
import pandas as pd
from functools import wraps
from math import ceil
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)

from utils.preprocessing import (
    load_dataset, fit_and_save_scaler, compute_elbow, find_optimal_k,
    train_kmeans, add_cluster_to_dataset, load_models, encode_new_case,
    normalize_new_case, get_normalization_info, FEATURE_COLS, compute_label,
    DATASET_PATH,
)
from utils.cbr_engine import run_cbr, retain
from utils.charts import plot_elbow, plot_clusters
from utils.expert import (
    login_expert, get_all_experts, add_expert, delete_expert,
    update_expert_password, save_cbr_history, get_cbr_history,
    clear_cbr_history, revise_label, delete_case, add_manual_case,
    get_label_stats,
)

app = Flask(__name__)
app.secret_key = 'cbr_kmeans_expert_secret_2024'

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    'df': None,
    'df_cluster': None,
    'scaler': None,
    'kmeans': None,
    'sse_list': [],
    'optimal_k': 3,
    'elbow_img': None,
    'cluster_img': None,
    'norm_info': {},
    'norm_sample': [],
    'trained': False,
}


def initialize_system():
    df = load_dataset()
    scaler, kmeans = load_models()
    sse_list = compute_elbow(df)
    optimal_k = find_optimal_k(sse_list, default_k=3)
    if scaler is None or kmeans is None:
        scaler = fit_and_save_scaler(df)
        kmeans = train_kmeans(df, optimal_k)
    else:
        optimal_k = kmeans.n_clusters
    df_cluster = add_cluster_to_dataset(df, kmeans)
    norm_info, norm_sample = get_normalization_info(df, scaler)
    elbow_img = plot_elbow(sse_list, optimal_k)
    cluster_img = plot_clusters(df_cluster, FEATURE_COLS, optimal_k)
    _state.update({
        'df': df, 'df_cluster': df_cluster, 'scaler': scaler,
        'kmeans': kmeans, 'sse_list': sse_list, 'optimal_k': optimal_k,
        'elbow_img': elbow_img, 'cluster_img': cluster_img,
        'norm_info': norm_info, 'norm_sample': norm_sample, 'trained': True,
    })


def _reload_df():
    df = load_dataset()
    df_cluster = add_cluster_to_dataset(df, _state['kmeans'])
    _state['df'] = df
    _state['df_cluster'] = df_cluster


# ── Auth decorator ────────────────────────────────────────────────────────────

def expert_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('expert_logged_in'):
            return redirect(url_for('expert_login'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', trained=_state['trained'],
                           optimal_k=_state['optimal_k'])


@app.route('/train', methods=['POST'])
def train():
    k = int(request.form.get('k', _state['optimal_k']))
    df = load_dataset()
    scaler = fit_and_save_scaler(df)
    kmeans = train_kmeans(df, k)
    df_cluster = add_cluster_to_dataset(df, kmeans)
    sse_list = compute_elbow(df)
    norm_info, norm_sample = get_normalization_info(df, scaler)
    elbow_img = plot_elbow(sse_list, k)
    cluster_img = plot_clusters(df_cluster, FEATURE_COLS, k)
    _state.update({
        'df': df, 'df_cluster': df_cluster, 'scaler': scaler,
        'kmeans': kmeans, 'sse_list': sse_list, 'optimal_k': k,
        'elbow_img': elbow_img, 'cluster_img': cluster_img,
        'norm_info': norm_info, 'norm_sample': norm_sample, 'trained': True,
    })
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    if not _state['trained']:
        return redirect(url_for('index'))
    df = _state['df_cluster']
    cluster_counts = df['cluster'].value_counts().sort_index().to_dict()
    label_counts = df['label_sentimen'].value_counts().to_dict()
    return render_template('dashboard.html', trained=True,
                           optimal_k=_state['optimal_k'],
                           sse_list=_state['sse_list'],
                           elbow_img=_state['elbow_img'],
                           cluster_img=_state['cluster_img'],
                           norm_info=_state['norm_info'],
                           norm_sample=_state['norm_sample'],
                           feature_cols=FEATURE_COLS,
                           cluster_counts=cluster_counts,
                           label_counts=label_counts,
                           total_data=len(df))


@app.route('/cbr', methods=['GET', 'POST'])
def cbr():
    """
    Alur CBR publik:
    - POST action=predict  → Retrieve + Reuse + Revise (otomatis)
    - POST action=retain   → simpan kasus ke CSV
    """
    if not _state['trained']:
        return redirect(url_for('index'))
    result = None
    form_data = {}

    if request.method == 'POST':
        action = request.form.get('action', 'predict')
        form_data = {
            'sex':       request.form.get('sex', 'M'),
            'address':   request.form.get('address', 'U'),
            'Pstatus':   request.form.get('Pstatus', 'T'),
            'studytime': float(request.form.get('studytime', 2)),
            'failures':  float(request.form.get('failures', 0)),
            'romantic':  request.form.get('romantic', 'No'),
            'famrel':    float(request.form.get('famrel', 3)),
            'freetime':  float(request.form.get('freetime', 3)),
            'goout':     float(request.form.get('goout', 3)),
            'absences':  float(request.form.get('absences', 0)),
        }
        dalc = float(request.form.get('dalc', 1))
        walc = float(request.form.get('walc', 1))

        if action == 'retain':
            saved = session.get('last_cbr_case')
            if saved:
                retain(saved['normalized_dict'], saved['kategori'], saved['cluster'])
                _reload_df()
                save_cbr_history({
                    'type': 'retain',
                    'input': saved['normalized_dict'],
                    'kategori': saved['kategori'],
                    'cluster': saved['cluster'],
                    'revisi_pakar': saved.get('revisi_pakar', False),
                    'user': 'Pengguna',
                })
                result = {'retained': True, 'message': 'Kasus berhasil disimpan ke basis kasus!'}
            return render_template('cbr.html', result=result, form_data=form_data,
                                   feature_cols=FEATURE_COLS, trained=True)

        # ENCODE + NORMALIZE
        encoded       = encode_new_case(form_data)
        normalized_arr = normalize_new_case(encoded)
        normalized_dict = {col: round(float(normalized_arr[i]), 6)
                           for i, col in enumerate(FEATURE_COLS)}

        # Cluster
        new_cluster = int(_state['kmeans'].predict(normalized_arr.reshape(1, -1))[0])

        # Label dari Dalc/Walc
        kategori_dalc, score, t1, t2 = compute_label(dalc, walc)

        # CBR pipeline
        cbr_result = run_cbr(
            df_with_cluster=_state['df_cluster'],
            new_case_normalized=normalized_arr,
            new_cluster=new_cluster,
            new_case_normalized_dict=normalized_dict,
            kategori_prediksi=kategori_dalc,
        )

        session['last_cbr_case'] = {
            'normalized_dict': normalized_dict,
            'kategori': cbr_result['revise']['kategori'],
            'cluster': new_cluster,
            'revisi_pakar': False,
        }

        save_cbr_history({
            'type': 'prediksi',
            'input_raw': form_data,
            'normalized': normalized_dict,
            'cluster': new_cluster,
            'kategori': cbr_result['revise']['kategori'],
            'similarity': cbr_result['revise']['similarity_score'],
            'warning': cbr_result['revise']['warning'],
            'user': 'Pengguna',
        })

        result = {
            'form_data': form_data, 'encoded': encoded,
            'normalized': normalized_dict, 'cluster': new_cluster,
            'cbr': cbr_result, 'dalc': dalc, 'walc': walc,
            'score_dalc': round(score, 4),
            'threshold1': round(t1, 4), 'threshold2': round(t2, 4),
            'kategori_input': kategori_dalc, 'retained': False,
        }

    return render_template('cbr.html', result=result, form_data=form_data,
                           feature_cols=FEATURE_COLS, trained=True,
                           optimal_k=_state['optimal_k'])


@app.route('/dataset')
def dataset_view():
    if not _state['trained']:
        return redirect(url_for('index'))
    df = _state['df_cluster']
    records = df.round(4).to_dict(orient='records')
    label_counts = df['label_sentimen'].value_counts().to_dict()
    return render_template('dataset.html', records=records,
                           columns=list(df.columns),
                           label_counts=label_counts,
                           total=len(df), trained=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT AUTH
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/pakar/login', methods=['GET', 'POST'])
def expert_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        expert = login_expert(username, password)
        if expert:
            session['expert_logged_in'] = True
            session['expert_username']  = expert['username']
            session['expert_nama']      = expert['nama']
            session['expert_jabatan']   = expert['jabatan']
            return redirect(url_for('expert_dashboard'))
        error = 'Username atau password salah.'
    return render_template('expert/login.html', error=error)


@app.route('/pakar/logout')
def expert_logout():
    for k in ('expert_logged_in', 'expert_username', 'expert_nama',
              'expert_jabatan', 'expert_cbr_case'):
        session.pop(k, None)
    return redirect(url_for('expert_login'))


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT CBR — dengan proses REVISE interaktif
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/pakar/cbr', methods=['GET', 'POST'])
@expert_required
def expert_cbr():
    """
    Alur CBR dari sisi pakar — 3 tahap:
      Tahap 1 (GET/POST action=input)  → Form input kasus baru
      Tahap 2 (POST action=reuse)      → Retrieve + Reuse → tampilkan Top-5
      Tahap 3 (POST action=revise)     → REVISE: pakar evaluasi & opsional override kategori
      Tahap 4 (POST action=retain)     → RETAIN: simpan ke CSV
    """
    if not _state['trained']:
        return redirect(url_for('expert_dashboard'))

    stage  = request.args.get('stage', 'input')
    result = None
    form_data = {}

    if request.method == 'POST':
        action = request.form.get('action', 'reuse')

        # ── Tahap 2: RETRIEVE + REUSE ──────────────────────────────────────
        if action == 'reuse':
            form_data = {
                'sex':       request.form.get('sex', 'M'),
                'address':   request.form.get('address', 'U'),
                'Pstatus':   request.form.get('Pstatus', 'T'),
                'studytime': float(request.form.get('studytime', 2)),
                'failures':  float(request.form.get('failures', 0)),
                'romantic':  request.form.get('romantic', 'No'),
                'famrel':    float(request.form.get('famrel', 3)),
                'freetime':  float(request.form.get('freetime', 3)),
                'goout':     float(request.form.get('goout', 3)),
                'absences':  float(request.form.get('absences', 0)),
            }
            dalc = float(request.form.get('dalc', 1))
            walc = float(request.form.get('walc', 1))

            encoded        = encode_new_case(form_data)
            normalized_arr = normalize_new_case(encoded)
            normalized_dict = {col: round(float(normalized_arr[i]), 6)
                               for i, col in enumerate(FEATURE_COLS)}
            new_cluster = int(_state['kmeans'].predict(normalized_arr.reshape(1, -1))[0])
            kategori_dalc, score, t1, t2 = compute_label(dalc, walc)
            cbr_result = run_cbr(
                df_with_cluster=_state['df_cluster'],
                new_case_normalized=normalized_arr,
                new_cluster=new_cluster,
                new_case_normalized_dict=normalized_dict,
                kategori_prediksi=kategori_dalc,
            )

            # Simpan state ke session untuk tahap berikutnya
            session['expert_cbr_case'] = {
                'form_data':       form_data,
                'encoded':         encoded,
                'normalized_dict': normalized_dict,
                'cluster':         new_cluster,
                'dalc': dalc, 'walc': walc,
                'score_dalc':  round(score, 4),
                'threshold1':  round(t1, 4),
                'threshold2':  round(t2, 4),
                'kategori_input': kategori_dalc,
                'cbr': {
                    'retrieve': cbr_result['retrieve'],
                    'reuse':    cbr_result['reuse'],
                    'revise':   cbr_result['revise'],
                },
            }

            save_cbr_history({
                'type': 'prediksi',
                'input_raw': form_data,
                'normalized': normalized_dict,
                'cluster': new_cluster,
                'kategori': cbr_result['revise']['kategori'],
                'similarity': cbr_result['revise']['similarity_score'],
                'warning': cbr_result['revise']['warning'],
                'user': session['expert_username'],
            })

            return redirect(url_for('expert_cbr', stage='revise'))

        # ── Tahap 3: REVISE oleh pakar ─────────────────────────────────────
        elif action == 'revise':
            saved = session.get('expert_cbr_case', {})
            if not saved:
                return redirect(url_for('expert_cbr', stage='input'))

            # Kategori yang dipilih pakar (bisa sama atau berbeda)
            kategori_pakar = request.form.get('kategori_final', '')
            catatan_pakar  = request.form.get('catatan_pakar', '').strip()
            kategori_sistem = saved['cbr']['revise']['kategori']

            direvisi = (kategori_pakar != kategori_sistem) and (kategori_pakar != '')
            kategori_final = kategori_pakar if kategori_pakar else kategori_sistem

            # Update session dengan keputusan pakar
            saved['kategori_final']  = kategori_final
            saved['kategori_sistem'] = kategori_sistem
            saved['direvisi']        = direvisi
            saved['catatan_pakar']   = catatan_pakar
            session['expert_cbr_case'] = saved

            save_cbr_history({
                'type': 'revise_pakar',
                'pakar': session['expert_username'],
                'kategori_sistem': kategori_sistem,
                'kategori_final':  kategori_final,
                'direvisi':        direvisi,
                'catatan':         catatan_pakar,
                'cluster':         saved['cluster'],
                'similarity':      saved['cbr']['revise']['similarity_score'],
            })

            return redirect(url_for('expert_cbr', stage='retain'))

        # ── Tahap 4: RETAIN ────────────────────────────────────────────────
        elif action == 'retain':
            saved = session.get('expert_cbr_case', {})
            if not saved:
                return redirect(url_for('expert_cbr', stage='input'))

            kategori_final = saved.get('kategori_final', saved['cbr']['revise']['kategori'])
            retain(saved['normalized_dict'], kategori_final, saved['cluster'])
            _reload_df()

            save_cbr_history({
                'type': 'retain',
                'pakar': session['expert_username'],
                'input': saved['normalized_dict'],
                'kategori': kategori_final,
                'direvisi': saved.get('direvisi', False),
                'catatan': saved.get('catatan_pakar', ''),
                'cluster': saved['cluster'],
            })

            # Bersihkan session CBR
            session.pop('expert_cbr_case', None)
            return redirect(url_for('expert_cbr', stage='done'))

        elif action == 'batal':
            session.pop('expert_cbr_case', None)
            return redirect(url_for('expert_cbr', stage='input'))

    # ── Render berdasarkan stage ────────────────────────────────────────────
    saved = session.get('expert_cbr_case')

    return render_template(
        'expert/cbr_pakar.html',
        stage=stage,
        saved=saved,
        feature_cols=FEATURE_COLS,
        optimal_k=_state['optimal_k'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT PANEL ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/pakar/dashboard')
@expert_required
def expert_dashboard():
    stats  = get_label_stats(DATASET_PATH)
    history = get_cbr_history(limit=10)
    df     = _state['df_cluster']
    cluster_counts = df['cluster'].value_counts().sort_index().to_dict()
    return render_template('expert/dashboard.html',
                           stats=stats, history=history,
                           cluster_counts=cluster_counts,
                           optimal_k=_state['optimal_k'],
                           total_data=len(df))


@app.route('/pakar/basis-kasus', methods=['GET', 'POST'])
@expert_required
def expert_basis_kasus():
    msg = None
    msg_type = 'success'
    if request.method == 'POST':
        action   = request.form.get('action')
        username = session['expert_username']

        if action == 'revisi_label':
            row_index = int(request.form.get('row_index'))
            new_label = request.form.get('new_label')
            ok, msg = revise_label(DATASET_PATH, row_index, new_label, username)
            msg_type = 'success' if ok else 'danger'
            _reload_df()

        elif action == 'hapus_kasus':
            row_index = int(request.form.get('row_index'))
            ok, msg = delete_case(DATASET_PATH, row_index, username)
            msg_type = 'success' if ok else 'danger'
            _reload_df()

        elif action == 'tambah_kasus':
            raw = {
                'sex':       request.form.get('sex'),
                'address':   request.form.get('address'),
                'Pstatus':   request.form.get('Pstatus'),
                'studytime': float(request.form.get('studytime', 2)),
                'failures':  float(request.form.get('failures', 0)),
                'romantic':  request.form.get('romantic'),
                'famrel':    float(request.form.get('famrel', 3)),
                'freetime':  float(request.form.get('freetime', 3)),
                'goout':     float(request.form.get('goout', 3)),
                'absences':  float(request.form.get('absences', 0)),
            }
            label   = request.form.get('label_sentimen', 'Normal')
            encoded = encode_new_case(raw)
            norm    = normalize_new_case(encoded)
            row = {col: round(float(norm[i]), 6) for i, col in enumerate(FEATURE_COLS)}
            row['label_sentimen'] = label
            ok, msg = add_manual_case(DATASET_PATH, row, username)
            msg_type = 'success' if ok else 'danger'
            _reload_df()

    df = _state['df_cluster'].copy()

    # Data terbaru di atas
    df = df.iloc[::-1].reset_index(drop=True)

    # Pagination
    PER_PAGE = 10
    page = request.args.get('page', 1, type=int)

    total = len(df)
    total_pages = ceil(total / PER_PAGE)

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE

    records = (
        df.iloc[start:end]
        .round(4)
        .to_dict(orient='records')
    )

    return render_template('expert/basis_kasus.html',
        records=records,
        columns=list(df.columns),
        total=total,
        page=page,
        total_pages=total_pages,
        per_page=PER_PAGE,
        msg=msg,
        msg_type=msg_type
    )


@app.route('/pakar/riwayat')
@expert_required
def expert_riwayat():
    history = get_cbr_history(limit=100)
    return render_template('expert/riwayat.html', history=history)


@app.route('/pakar/riwayat/clear', methods=['POST'])
@expert_required
def expert_clear_riwayat():
    clear_cbr_history()
    return redirect(url_for('expert_riwayat'))


@app.route('/pakar/akun', methods=['GET', 'POST'])
@expert_required
def expert_akun():
    msg = None
    msg_type = 'success'
    is_admin = session.get('expert_username') == 'admin'

    if request.method == 'POST' and is_admin:
        action = request.form.get('action')
        if action == 'tambah':
            ok, msg = add_expert(
                request.form.get('username', '').strip(),
                request.form.get('password', '').strip(),
                request.form.get('nama', '').strip(),
                request.form.get('jabatan', '').strip(),
            )
            msg_type = 'success' if ok else 'danger'
        elif action == 'hapus':
            ok, msg = delete_expert(request.form.get('target_username'))
            msg_type = 'success' if ok else 'danger'

    if request.method == 'POST' and not is_admin:
        action = request.form.get('action')
        if action == 'ganti_password':
            ok, msg = update_expert_password(
                session['expert_username'],
                request.form.get('new_password', '').strip(),
            )
            msg_type = 'success' if ok else 'danger'

    experts = get_all_experts()
    return render_template('expert/akun.html',
                           experts=experts, is_admin=is_admin,
                           msg=msg, msg_type=msg_type)


@app.route('/pakar/retrain', methods=['POST'])
@expert_required
def expert_retrain():
    k  = int(request.form.get('k', _state['optimal_k']))
    df = load_dataset()
    scaler  = fit_and_save_scaler(df)
    kmeans  = train_kmeans(df, k)
    df_cluster = add_cluster_to_dataset(df, kmeans)
    sse_list   = compute_elbow(df)
    norm_info, norm_sample = get_normalization_info(df, scaler)
    elbow_img   = plot_elbow(sse_list, k)
    cluster_img = plot_clusters(df_cluster, FEATURE_COLS, k)
    _state.update({
        'df': df, 'df_cluster': df_cluster, 'scaler': scaler,
        'kmeans': kmeans, 'sse_list': sse_list, 'optimal_k': k,
        'elbow_img': elbow_img, 'cluster_img': cluster_img,
        'norm_info': norm_info, 'norm_sample': norm_sample, 'trained': True,
    })
    save_cbr_history({
        'type': 're-train',
        'pakar': session['expert_username'],
        'k': k, 'total_data': len(df),
    })
    return redirect(url_for('expert_dashboard'))


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=5000)
