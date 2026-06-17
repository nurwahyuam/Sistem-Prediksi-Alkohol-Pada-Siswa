"""
app.py
======
Entry point utama aplikasi Flask.

CBR Alcohol - Sistem Prediksi Tingkat Konsumsi Alkohol Siswa
Menggunakan: K-Means Clustering + Cosine Similarity + Case Based Reasoning

Routes:
    GET  /                          → Dashboard
    GET  /preprocessing             → Halaman preprocessing
    POST /preprocessing/run         → Jalankan preprocessing
    GET  /clustering                → Halaman clustering
    POST /clustering/run            → Jalankan training KMeans
    GET  /predict                   → Form input kasus baru
    POST /predict                   → Proses prediksi
    GET  /result/<session_id>       → Halaman hasil retrieve + reuse
    GET  /expert/<session_id>       → Ruang pakar (revise)
    POST /expert/save               → Simpan keputusan pakar (retain)
    GET  /casebase                  → Tabel case base
    GET  /history                   → Riwayat retrieve
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import json
import uuid
from datetime import datetime

# Import services
from services.preprocessing import (
    run_preprocessing, load_case_base, cek_preprocessing_selesai,
    normalisasi_satu_kasus, load_preprocessed
)
from services.clustering import (
    run_clustering, prediksi_cluster_baru, cek_training_selesai,
    load_cluster_info
)
from services.retrieve import run_retrieve
from services.reuse import run_reuse
from services.revise import simpan_revisi, load_revision_log
from services.retain import run_retain, load_retrieve_history
from services.evaluation import run_evaluation
from services.auth import verify_login, login_required, is_logged_in

# ===== INISIALISASI FLASK =====
app = Flask(__name__)
app.secret_key = 'cbr_alcohol_secret_key_2024'

# Penyimpanan sementara sesi prediksi (in-memory, bukan DB)
# Key: session_id, Value: dict data prediksi
PREDICT_SESSIONS = {}


@app.context_processor
def inject_auth_context():
    """Sediakan status login pakar (is_expert, expert_name) ke semua template."""
    return dict(is_expert=is_logged_in(), expert_name=session.get('expert_name'))


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_stats_dashboard():
    """Ambil statistik untuk dashboard."""
    stats = {
        'preprocessing_selesai': cek_preprocessing_selesai(),
        'training_selesai': cek_training_selesai(),
        'total_case_base': 0,
        'total_normal': 0,
        'total_waspada': 0,
        'total_bahaya': 0,
        'total_history': 0,
        'k_optimal': '-',
    }

    df_case = load_case_base()
    if df_case is not None:
        stats['total_case_base'] = len(df_case)
        label_counts = df_case['label'].value_counts().to_dict()
        stats['total_normal'] = label_counts.get('Normal', 0)
        stats['total_waspada'] = label_counts.get('Waspada', 0)
        stats['total_bahaya'] = label_counts.get('Bahaya', 0)

    df_cluster = load_cluster_info()
    if df_cluster is not None and len(df_cluster) > 0:
        stats['k_optimal'] = int(df_cluster['k_optimal'].iloc[0])

    df_history = load_retrieve_history()
    if df_history is not None:
        stats['total_history'] = len(df_history)

    return stats


# ===================================================================
# ROUTES - DASHBOARD
# ===================================================================

@app.route('/')
def index():
    """Dashboard utama."""
    stats = get_stats_dashboard()
    return render_template('index.html', stats=stats)


# ===================================================================
# ROUTES - AUTENTIKASI PAKAR
# ===================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login Pakar menggunakan Flask Session."""
    if is_logged_in():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        expert_name = verify_login(username, password)

        if expert_name:
            session['role'] = 'pakar'
            session['expert_name'] = expert_name
            flash(f"Selamat datang, {expert_name}!", 'success')
            next_url = request.args.get('next') or url_for('index')
            return redirect(next_url)

        flash('Username atau password salah.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout Pakar, hapus session."""
    session.pop('role', None)
    session.pop('expert_name', None)
    flash('Anda telah logout dari mode Pakar.', 'info')
    return redirect(url_for('index'))


# ===================================================================
# ROUTES - PREPROCESSING
# ===================================================================

@app.route('/preprocessing')
def preprocessing():
    """Halaman preprocessing dataset."""
    sudah_diproses = cek_preprocessing_selesai()
    df_preview = None

    if sudah_diproses:
        df = load_preprocessed()
        if df is not None:
            # Ambil 10 data pertama untuk preview
            df_preview = df.head(10).to_dict('records')

    return render_template('preprocessing.html',
                           sudah_diproses=sudah_diproses,
                           df_preview=df_preview)


@app.route('/preprocessing/run', methods=['POST'])
def preprocessing_run():
    """Jalankan preprocessing dataset."""
    try:
        hasil = run_preprocessing()
        flash(f"Preprocessing berhasil! {hasil['total_data']} data diproses.", 'success')
        flash(f"Distribusi label: Normal={hasil['distribusi_label'].get('Normal',0)}, "
              f"Waspada={hasil['distribusi_label'].get('Waspada',0)}, "
              f"Bahaya={hasil['distribusi_label'].get('Bahaya',0)}", 'info')
        session['preprocessing_result'] = hasil
    except Exception as e:
        flash(f"Error saat preprocessing: {str(e)}", 'danger')

    return redirect(url_for('preprocessing'))


# ===================================================================
# ROUTES - CLUSTERING
# ===================================================================

@app.route('/clustering')
def clustering():
    """Halaman training K-Means."""
    if not cek_preprocessing_selesai():
        flash('Preprocessing harus dilakukan terlebih dahulu!', 'warning')
        return redirect(url_for('preprocessing'))

    training_selesai = cek_training_selesai()
    cluster_info = None
    elbow_data = None

    if training_selesai:
        df_info = load_cluster_info()
        if df_info is not None:
            cluster_info = df_info.to_dict('records')

    # Ambil data elbow dari session jika ada
    if 'elbow_data' in session:
        elbow_data = session.get('elbow_data')

    return render_template('clustering.html',
                           training_selesai=training_selesai,
                           cluster_info=cluster_info,
                           elbow_data=elbow_data)


@app.route('/clustering/run', methods=['POST'])
def clustering_run():
    """Jalankan training K-Means."""
    try:
        hasil = run_clustering()

        # Simpan data elbow ke session untuk ditampilkan di chart
        session['elbow_data'] = {
            'k_values': hasil['k_values'],
            'sse_values': hasil['sse_values'],
            'k_optimal': hasil['k_optimal']
        }

        flash(f"Training berhasil! K Optimal = {hasil['k_optimal']} dari {hasil['total_data']} data.", 'success')

        for info in hasil['cluster_info']:
            flash(f"Cluster {info['cluster_id']}: {info['total_data']} data, "
                  f"dominan '{info['dominant_label']}'", 'info')

    except Exception as e:
        flash(f"Error saat training: {str(e)}", 'danger')

    return redirect(url_for('clustering'))


# ===================================================================
# ROUTES - PREDIKSI KASUS BARU
# ===================================================================

@app.route('/predict')
def predict():
    """Form input kasus baru."""
    if not cek_training_selesai():
        flash('Training K-Means harus dilakukan terlebih dahulu!', 'warning')
        return redirect(url_for('clustering'))

    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict_post():
    """Proses prediksi kasus baru."""
    try:
        # 1. Ambil input dari form
        input_raw = {
            'sex': request.form.get('sex', 'F'),
            'address': request.form.get('address', 'U'),
            'pstatus': request.form.get('pstatus', 'T'),
            'studytime': float(request.form.get('studytime', 2)),
            'failures': float(request.form.get('failures', 0)),
            'romantic': request.form.get('romantic', 'no'),
            'famrel': float(request.form.get('famrel', 3)),
            'freetime': float(request.form.get('freetime', 3)),
            'goout': float(request.form.get('goout', 3)),
            'absences': float(request.form.get('absences', 0)),
        }

        # 2. Preprocessing kasus baru (encoding + normalisasi)
        fitur_normalized = normalisasi_satu_kasus(input_raw)

        # 3. Prediksi cluster
        cluster_id = prediksi_cluster_baru(fitur_normalized)

        # 4. RETRIEVE - Hitung cosine similarity
        retrieve_result = run_retrieve(fitur_normalized, cluster_id, top_n=5)

        # 5. REUSE - Ambil prediksi dari kasus terbaik
        reuse_result = run_reuse(retrieve_result['top_cases'])

        # 6. Hitung alcohol_score untuk kasus baru
        # (untuk kasus baru, kita tidak tahu Dalc/Walc, jadi estimasi dari label)
        # Gunakan nilai tengah label sebagai estimasi
        alcohol_score_est = {
            'Normal': 1.67,
            'Waspada': 3.0,
            'Bahaya': 4.34
        }.get(reuse_result['prediksi_label'], 2.5)

        # 7. Simpan ke session sementara
        session_id = str(uuid.uuid4())[:8]
        PREDICT_SESSIONS[session_id] = {
            'input_raw': input_raw,
            'fitur_normalized': fitur_normalized,
            'cluster_id': cluster_id,
            'retrieve_result': retrieve_result,
            'reuse_result': reuse_result,
            'alcohol_score_est': alcohol_score_est,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return redirect(url_for('result', session_id=session_id))

    except Exception as e:
        flash(f"Error saat prediksi: {str(e)}", 'danger')
        return redirect(url_for('predict'))


# ===================================================================
# ROUTES - HASIL RETRIEVE & REUSE
# ===================================================================

@app.route('/result/<session_id>')
def result(session_id):
    """Halaman hasil retrieve dan reuse."""
    if session_id not in PREDICT_SESSIONS:
        flash('Sesi prediksi tidak ditemukan. Silakan input kasus baru.', 'warning')
        return redirect(url_for('predict'))

    data = PREDICT_SESSIONS[session_id]
    return render_template('result.html',
                           session_id=session_id,
                           input_raw=data['input_raw'],
                           fitur_normalized=data['fitur_normalized'],
                           cluster_id=data['cluster_id'],
                           retrieve_result=data['retrieve_result'],
                           reuse_result=data['reuse_result'],
                           timestamp=data['timestamp'])


# ===================================================================
# ROUTES - RUANG PAKAR (REVISE)
# ===================================================================

@app.route('/expert/<session_id>')
@login_required
def expert(session_id):
    """Ruang pakar untuk revisi prediksi."""
    if session_id not in PREDICT_SESSIONS:
        flash('Sesi prediksi tidak ditemukan.', 'warning')
        return redirect(url_for('predict'))

    data = PREDICT_SESSIONS[session_id]
    label_options = ['Normal', 'Waspada', 'Bahaya']

    return render_template('expert.html',
                           session_id=session_id,
                           input_raw=data['input_raw'],
                           fitur_normalized=data['fitur_normalized'],
                           cluster_id=data['cluster_id'],
                           retrieve_result=data['retrieve_result'],
                           reuse_result=data['reuse_result'],
                           label_options=label_options,
                           timestamp=data['timestamp'])


@app.route('/expert/save', methods=['POST'])
@login_required
def expert_save():
    """Simpan keputusan pakar dan jalankan Retain."""
    session_id = request.form.get('session_id')

    if session_id not in PREDICT_SESSIONS:
        flash('Sesi prediksi tidak ditemukan.', 'warning')
        return redirect(url_for('predict'))

    data = PREDICT_SESSIONS[session_id]

    expert_label = request.form.get('expert_label')
    note = request.form.get('note', '')
    system_label = data['reuse_result']['prediksi_label']

    try:
        # Generate case_id sementara untuk log revisi
        temp_case_id = f"TEMP-{session_id}"

        # REVISE - Simpan ke revision_log
        revise_result = simpan_revisi(
            case_id=temp_case_id,
            system_label=system_label,
            final_label=expert_label,
            note=note,
            expert_name=session.get('expert_name', 'pakar')
        )

        # RETAIN - Simpan kasus ke case base
        retain_result = run_retain(
            fitur_normalized_dict=data['fitur_normalized'],
            cluster_id=data['cluster_id'],
            alcohol_score=data['alcohol_score_est'],
            system_label=system_label,
            final_label=expert_label,
            top_cases=data['retrieve_result']['top_cases']
        )

        # Hapus sesi setelah retain berhasil
        del PREDICT_SESSIONS[session_id]

        if revise_result['label_berubah']:
            flash(f"Label direvisi: '{system_label}' → '{expert_label}'. "
                  f"Kasus disimpan sebagai {retain_result['case_id']} (EXPERT_REVISED).", 'warning')
        else:
            flash(f"Label dikonfirmasi: '{expert_label}'. "
                  f"Kasus disimpan sebagai {retain_result['case_id']} (VALIDATED).", 'success')

        flash(retain_result['pesan'], 'info')

    except Exception as e:
        flash(f"Error saat menyimpan: {str(e)}", 'danger')
        return redirect(url_for('expert', session_id=session_id))

    return redirect(url_for('casebase'))


# ===================================================================
# ROUTES - CASE BASE
# ===================================================================

@app.route('/casebase')
def casebase():
    """Tampilkan tabel case base."""
    df = load_case_base()

    if df is None:
        flash('Case base belum tersedia. Jalankan preprocessing terlebih dahulu.', 'warning')
        return render_template('casebase.html', cases=[], total=0)

    # Urutkan berdasarkan created_at descending
    if 'created_at' in df.columns:
        df = df.sort_values('created_at', ascending=False)

    cases = df.to_dict('records')

    # Statistik
    stats = {
        'total': len(cases),
        'training': len([c for c in cases if c.get('status') == 'TRAINING']),
        'validated': len([c for c in cases if c.get('status') == 'VALIDATED']),
        'expert_revised': len([c for c in cases if c.get('status') == 'EXPERT_REVISED']),
        'normal': len([c for c in cases if c.get('label') == 'Normal']),
        'waspada': len([c for c in cases if c.get('label') == 'Waspada']),
        'bahaya': len([c for c in cases if c.get('label') == 'Bahaya']),
    }

    return render_template('casebase.html', cases=cases, stats=stats)


# ===================================================================
# ROUTES - RIWAYAT
# ===================================================================

@app.route('/history')
def history():
    """Tampilkan riwayat retrieve."""
    df_history = load_retrieve_history()
    df_revisi = load_revision_log()

    history_list = []
    if df_history is not None and len(df_history) > 0:
        history_list = df_history.sort_values('created_at', ascending=False).to_dict('records')

    revisi_list = []
    if df_revisi is not None and len(df_revisi) > 0:
        revisi_list = df_revisi.sort_values('created_at', ascending=False).to_dict('records')

    return render_template('history.html',
                           history_list=history_list,
                           revisi_list=revisi_list)


# ===================================================================
# ROUTES - EVALUASI SISTEM (KHUSUS PAKAR)
# ===================================================================

@app.route('/evaluation')
@login_required
def evaluation():
    """Halaman evaluasi sistem CBR (split 80:20) — khusus Pakar."""
    hasil = None
    cases = []
    try:
        hasil = run_evaluation()
        df_case = load_case_base()
        if df_case is not None:
            cases = df_case.to_dict('records')
    except Exception as e:
        flash(f"Error saat evaluasi: {str(e)}", 'danger')

    return render_template('evaluation.html', hasil=hasil, cases=cases)


# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    # Pastikan folder models ada
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    app.run(debug=True, host='localhost', port=5000)
