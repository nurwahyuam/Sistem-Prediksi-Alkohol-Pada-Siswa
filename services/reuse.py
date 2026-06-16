"""
reuse.py
========
Modul CBR - Tahap REUSE

Alur:
1. Terima hasil dari Retrieve (Top 5 kasus)
2. Ambil kasus dengan similarity TERTINGGI
3. Gunakan label kasus tersebut sebagai prediksi sistem
4. Return prediksi beserta detail kasus referensi
"""


def run_reuse(top_cases):
    """
    Jalankan proses Reuse CBR.
    
    Parameters:
        top_cases: list dict dari hasil retrieve (sudah diurutkan similarity tertinggi)
    
    Returns:
        dict berisi:
        - prediksi_label: label prediksi sistem
        - kasus_referensi: dict data kasus yang paling mirip
        - similarity: nilai similarity kasus referensi
        - penjelasan: teks penjelasan keputusan
    """
    if not top_cases:
        return {
            'prediksi_label': 'Tidak Diketahui',
            'kasus_referensi': None,
            'similarity': 0,
            'penjelasan': 'Tidak ada kasus serupa ditemukan dalam case base.'
        }

    # Ambil kasus dengan similarity tertinggi (sudah urut dari retrieve)
    kasus_terbaik = top_cases[0]

    prediksi = kasus_terbaik['label']
    similarity = kasus_terbaik['similarity']

    # Penjelasan keputusan
    penjelasan = (
        f"Sistem memprediksi label '{prediksi}' berdasarkan kasus {kasus_terbaik['case_id']} "
        f"yang memiliki tingkat kemiripan tertinggi sebesar {similarity:.4f} "
        f"({float(similarity)*100:.2f}%)."
    )

    # Tentukan badge warna berdasarkan label
    badge_map = {
        'Normal': 'success',
        'Waspada': 'warning',
        'Bahaya': 'danger'
    }
    badge_color = badge_map.get(prediksi, 'secondary')

    # Tingkat kepercayaan berdasarkan similarity
    if similarity >= 0.9:
        tingkat_kepercayaan = 'Sangat Tinggi'
    elif similarity >= 0.7:
        tingkat_kepercayaan = 'Tinggi'
    elif similarity >= 0.5:
        tingkat_kepercayaan = 'Sedang'
    else:
        tingkat_kepercayaan = 'Rendah'

    return {
        'prediksi_label': prediksi,
        'kasus_referensi': kasus_terbaik,
        'similarity': similarity,
        'similarity_pct': f"{float(similarity)*100:.2f}%",
        'badge_color': badge_color,
        'tingkat_kepercayaan': tingkat_kepercayaan,
        'penjelasan': penjelasan,
        'top_cases': top_cases  # semua top cases untuk ditampilkan
    }
