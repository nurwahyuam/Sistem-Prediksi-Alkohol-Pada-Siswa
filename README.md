# CBR KMeans — Sistem Klasifikasi Risiko Alkohol Siswa

Sistem web berbasis **Case Based Reasoning (CBR)** menggunakan
**K-Means Clustering** dan **Cosine Similarity** untuk mengklasifikasi
tingkat risiko konsumsi alkohol siswa.

---

## Struktur Proyek

```
cbr_project/
│
├── app.py                  # Flask main application
├── run.py                  # Entry point
├── requirements.txt
│
├── dataset/
│   └── student_dataset.csv # Basis kasus (bertambah setelah retain)
│
├── saved_model/
│   ├── scaler.pkl          # MinMaxScaler (dibuat otomatis)
│   └── kmeans.pkl          # KMeans model (dibuat otomatis)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # Encoding, normalisasi, elbow, train
│   ├── cbr_engine.py       # Retrieve, Reuse, Revise, Retain
│   └── charts.py           # Matplotlib chart → base64
│
├── templates/
│   ├── base.html           # Layout utama (sidebar, nav)
│   ├── index.html          # Halaman beranda
│   ├── dashboard.html      # Dashboard analisis
│   ├── cbr.html            # Halaman prediksi CBR
│   └── dataset.html        # Tampilan dataset
│
└── static/
    ├── css/
    ├── js/
    └── img/
```

---

## Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
python run.py
```

### 3. Buka Browser
```
http://127.0.0.1:5000
```

---

## Alur Sistem

### Tahap 1 — Load Dataset
- Baca `student_dataset.csv` (395 records, 10 fitur)

### Tahap 2 — Encoding
| Fitur | Mapping |
|-------|---------|
| Sex | F=0, M=1 |
| Address | R=0, U=1 |
| Pstatus | A=0, T=1 |
| Romantic | No=0, Yes=1 |

### Tahap 3 — Normalisasi Min-Max
```
X = (x - xmin) / (xmax - xmin)
```
Fitur: studytime, failures, famrel, freetime, goout, absences

### Tahap 4 — Labeling
```
score = 0.6 × Dalc + 0.4 × Walc
```
Equal Interval (range 1–5):
- **Normal**: score ≤ 2.33
- **Waspada**: 2.33 < score ≤ 3.67
- **Bahaya**: score > 3.67

### Tahap 5 — K-Means
- Elbow Method K=1..10
- Pilih K optimal (default K=3 jika elbow tidak jelas)
- Tambahkan kolom `cluster`

### Tahap 6 — CBR Pipeline
- **RETRIEVE**: Ambil kasus dengan cluster sama
- **REUSE**: Hitung cosine similarity semua kasus
- **REVISE**: Pilih similarity tertinggi, warning jika < 0.5
- **RETAIN**: Simpan kasus baru ke CSV

---

## Rumus Cosine Similarity
```
similarity = (A · B) / (||A|| × ||B||)
```

---

## Stack Teknologi
- **Backend**: Python Flask
- **ML**: scikit-learn (KMeans, MinMaxScaler)
- **Charts**: Matplotlib (base64 PNG)
- **UI**: Bootstrap 5.3 Dark Theme
- **Font**: Space Grotesk, JetBrains Mono
