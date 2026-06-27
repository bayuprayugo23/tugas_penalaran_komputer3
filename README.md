#Muhamad Alfian Bayu Prayugo - 202310370311204
#Afifuddin Fajrul Falah - 202310370311233
# Case-Based Reasoning Putusan PN Mojokerto - Pidana Umum Penipuan

Project ini dibuat untuk tugas **Penalaran Komputer SubCPMK-3**: implementasi sistem **Case-Based Reasoning (CBR)** sederhana berbasis Python untuk analisis putusan pengadilan.

## Identitas Project

- Domain hukum: **Pidana Umum - Penipuan**
- Pengadilan: **Pengadilan Negeri Mojokerto / PN Mojokerto**
- Target data: **45 putusan**
- Sumber data: Direktori Putusan Mahkamah Agung RI
- Metode retrieval utama: **TF-IDF + Cosine Similarity**
- Model klasifikasi tambahan: **Linear SVM** jika label hasil putusan lebih dari satu kelas

## Struktur Folder

```text
cbr_pn_mojokerto_penipuan/
├── data/
│   ├── raw/                  # hasil ekstraksi teks putusan: case_001.txt ... case_045.txt
│   ├── processed/            # cases.csv, cases.json, manifest_downloads.csv
│   ├── eval/                 # queries.json, retrieval_metrics.csv, prediction_metrics.csv
│   └── results/              # predictions.csv
├── logs/                     # cleaning.log
├── models/                   # vectorizer, matrix, model SVM
├── notebooks/                # notebook per tahap CBR
├── src/                      # helper config dan text processing
├── 01_scrape_cases.py        # Tahap 1: membangun case base
├── 02_representation.py      # Tahap 2: case representation
├── 03_retrieval.py           # Tahap 3: retrieval
├── 04_predict.py             # Tahap 4: solution reuse
├── 05_evaluation.py          # Tahap 5: evaluation
├── run_pipeline.py           # menjalankan semua tahap
├── requirements.txt
└── README.md
```

## Instalasi

### 1. Buat virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac/Google Colab:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install library

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Pipeline End-to-End

Jalankan perintah berikut dari folder utama project:

```bash
python run_pipeline.py --limit 45
```

Jika website Direktori MA lambat atau timeout, naikkan delay:

```bash
python run_pipeline.py --limit 45 --delay 3 --max-pages 40
```

Jika data raw sudah pernah diunduh dan ingin menjalankan ulang proses representation sampai evaluasi:

```bash
python run_pipeline.py --skip-scrape
```

## Cara Menjalankan Per Tahap

### Tahap 1 - Membangun Case Base

```bash
python 01_scrape_cases.py --limit 45 --max-pages 40 --delay 2
```

Output:

- `data/raw/case_001.txt` sampai `case_045.txt`
- `data/processed/manifest_downloads.csv`
- `logs/cleaning.log`

### Tahap 2 - Case Representation

```bash
python 02_representation.py
```

Output:

- `data/processed/cases.csv`
- `data/processed/cases.json`

Kolom penting:

- `case_id`
- `no_perkara`
- `pengadilan`
- `domain`
- `tanggal_putusan`
- `pasal`
- `pihak`
- `ringkasan_fakta`
- `argumen_hukum`
- `amar_putusan`
- `label_putusan`
- `solution_class`
- `word_count`
- `text_full`

### Tahap 3 - Case Retrieval

```bash
python 03_retrieval.py
```

Output:

- `models/tfidf_vectorizer.joblib`
- `models/tfidf_matrix.joblib`
- `models/case_index.csv`
- `data/eval/queries.json`
- `data/eval/classification_report.csv`

Contoh menjalankan retrieval manual:

```bash
python 03_retrieval.py --query "terdakwa melakukan penipuan dengan bujuk rayu dan korban menyerahkan uang" --k 5
```

### Tahap 4 - Case/Solution Reuse

```bash
python 04_predict.py --k 5
```

Output:

- `data/results/predictions.csv`

Contoh prediksi manual:

```bash
python 04_predict.py --query "korban menyerahkan uang karena dijanjikan pekerjaan, tetapi janji tidak dipenuhi" --k 5
```

### Tahap 5 - Model Evaluation

```bash
python 05_evaluation.py --k 5
```

Output:

- `data/eval/retrieval_metrics.csv`
- `data/eval/prediction_metrics.csv`
- `data/eval/retrieval_metrics_chart.png`

## Penjelasan Singkat Metode

### 1. Build Case Base

Sistem mengambil daftar putusan dari Direktori Putusan MA RI pada filter:

- `pengadilan/pn-mojokerto`
- `kategori/penipuan-1`

Setiap putusan diunduh, diekstrak ke teks, lalu dibersihkan dari header/footer, nomor halaman, watermark, dan spasi berlebih.

### 2. Case Representation

Setiap putusan direpresentasikan menjadi struktur data berisi metadata dan fitur teks, seperti nomor perkara, tanggal putusan, pasal, pihak, ringkasan fakta, argumen hukum, amar putusan, label, kelas solusi, dan jumlah kata.

### 3. Case Retrieval

Teks penting dari setiap putusan digabung, lalu diubah menjadi vektor menggunakan **TF-IDF**. Query kasus baru juga diubah menjadi vektor TF-IDF, kemudian dihitung kemiripannya dengan seluruh kasus memakai **cosine similarity**. Sistem mengembalikan top-k kasus termirip.

### 4. Case/Solution Reuse

Dari top-k kasus termirip, sistem mengambil `solution_class` dan `amar_putusan`. Prediksi solusi dipilih menggunakan **weighted similarity voting**, yaitu solusi dari kasus yang lebih mirip diberi bobot lebih besar.

### 5. Evaluation

Evaluasi retrieval dihitung menggunakan:

- Accuracy / Hit@K
- Precision@K
- Recall@K
- F1@K

Jika ground-truth solusi tersedia, prediksi solusi juga dievaluasi dengan classification report.

## Catatan untuk Upload GitHub

1. Buat repository public di GitHub.
2. Upload semua isi folder project ini.
3. Jalankan pipeline terlebih dahulu agar folder `data/raw`, `data/processed`, `data/eval`, dan `data/results` terisi.
4. Pastikan `README.md` dan `requirements.txt` ikut terupload.
5. Salin link repository ke LMS.

## Troubleshooting

### Website MA timeout

Jalankan ulang dengan delay lebih besar:

```bash
python run_pipeline.py --limit 45 --delay 4 --max-pages 50
```

### Data raw belum ada

Pastikan koneksi internet aktif, lalu jalankan:

```bash
python 01_scrape_cases.py --limit 45
```

### SVM tidak dilatih

Jika semua putusan memiliki label sama, SVM otomatis dilewati. Retrieval tetap berjalan karena metode utama menggunakan TF-IDF + cosine similarity.
