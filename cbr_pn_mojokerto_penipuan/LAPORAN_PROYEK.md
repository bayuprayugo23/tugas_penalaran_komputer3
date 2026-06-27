# Laporan Proyek CBR Putusan Pengadilan

## Judul
Implementasi Case-Based Reasoning untuk Analisis Putusan Pidana Umum - Penipuan pada Pengadilan Negeri Mojokerto

## 1. Deskripsi Proyek
Proyek ini merancang sistem Case-Based Reasoning (CBR) sederhana berbasis Python untuk membantu proses pencarian kasus putusan pengadilan yang mirip. Domain yang dipilih adalah **Pidana Umum - Penipuan** pada **Pengadilan Negeri Mojokerto** dengan target **45 putusan** dari Direktori Putusan Mahkamah Agung RI.

Sistem dibangun mengikuti siklus CBR, yaitu:

1. Membangun case base.
2. Merepresentasikan kasus.
3. Melakukan retrieval kasus serupa.
4. Melakukan reuse solusi dari kasus lama.
5. Melakukan evaluasi model.

## 2. Dataset
Dataset diperoleh dari Direktori Putusan Mahkamah Agung RI dengan filter:

- Kategori: Pidana Umum - Penipuan
- Pengadilan: PN Mojokerto
- Jumlah target: 45 putusan

File teks hasil ekstraksi disimpan dalam folder `data/raw/` dengan format `case_001.txt` sampai `case_045.txt`.

## 3. Tahap 1 - Membangun Case Base
Tahap ini dilakukan dengan script `01_scrape_cases.py`. Script mengambil halaman daftar putusan, menemukan link detail putusan, mencari link PDF/download jika tersedia, mengekstrak isi putusan menjadi teks, lalu melakukan pembersihan awal.

Pembersihan yang dilakukan:

- Menghapus header/footer umum Direktori Putusan MA.
- Menghapus nomor halaman.
- Menghapus watermark/disclaimer sederhana.
- Menormalisasi spasi dan karakter.
- Menyimpan hasil dalam format `.txt`.

Output tahap ini:

- `data/raw/case_001.txt` sampai `data/raw/case_045.txt`
- `data/processed/manifest_downloads.csv`
- `logs/cleaning.log`

## 4. Tahap 2 - Case Representation
Tahap ini dilakukan dengan script `02_representation.py`. Setiap file putusan direpresentasikan ke dalam struktur data tabular.

Metadata dan fitur yang diekstrak:

- `case_id`
- `no_perkara`
- `pengadilan`
- `domain`
- `tahun`
- `tanggal_register`
- `tanggal_putusan`
- `tanggal_upload`
- `pasal`
- `pihak`
- `ringkasan_fakta`
- `argumen_hukum`
- `amar_putusan`
- `label_putusan`
- `solution_class`
- `word_count`
- `text_full`

Output tahap ini:

- `data/processed/cases.csv`
- `data/processed/cases.json`

## 5. Tahap 3 - Case Retrieval
Tahap retrieval menggunakan pendekatan statistik **TF-IDF** dan **cosine similarity**. Teks yang digunakan sebagai representasi dokumen adalah gabungan dari ringkasan fakta, argumen hukum, pasal, amar putusan, dan kelas solusi.

Langkah retrieval:

1. Preprocessing query.
2. Mengubah query menjadi vektor TF-IDF.
3. Menghitung cosine similarity antara query dan semua kasus lama.
4. Mengambil top-k kasus dengan nilai similarity tertinggi.

Script utama:

- `03_retrieval.py`

Output:

- `models/tfidf_vectorizer.joblib`
- `models/tfidf_matrix.joblib`
- `models/case_index.csv`
- `data/eval/queries.json`
- `data/eval/classification_report.csv`

## 6. Tahap 4 - Case/Solution Reuse
Tahap reuse mengambil solusi dari top-k kasus termirip. Solusi diwakili oleh `solution_class` dan ringkasan `amar_putusan` dari kasus lama. Algoritma yang digunakan adalah **weighted similarity voting**, yaitu solusi dari kasus yang similarity-nya lebih tinggi memiliki bobot lebih besar.

Script utama:

- `04_predict.py`

Output:

- `data/results/predictions.csv`

## 7. Tahap 5 - Evaluasi Model
Evaluasi retrieval dilakukan menggunakan query evaluasi yang tersimpan dalam `data/eval/queries.json`.

Metrik yang digunakan:

- Accuracy / Hit@K
- Precision@K
- Recall@K
- F1@K

Script utama:

- `05_evaluation.py`

Output:

- `data/eval/retrieval_metrics.csv`
- `data/eval/prediction_metrics.csv`
- `data/eval/retrieval_metrics_chart.png`

## 8. Analisis Kegagalan Model
Kemungkinan kegagalan model retrieval dapat terjadi karena beberapa faktor:

1. Teks putusan hasil PDF tidak lengkap atau hasil ekstraksi kurang bersih.
2. Struktur putusan berbeda-beda sehingga regex metadata tidak selalu sempurna.
3. Query terlalu pendek sehingga tidak cukup merepresentasikan fakta kasus.
4. TF-IDF sensitif terhadap variasi kata, sehingga sinonim atau konteks semantik belum tertangkap secara optimal.
5. Label putusan tidak seimbang, misalnya mayoritas kasus berakhir terbukti.

## 9. Rekomendasi Perbaikan
Beberapa perbaikan yang dapat dilakukan:

1. Melakukan pengecekan manual terhadap teks yang char_count-nya rendah.
2. Menambah jumlah data lebih dari 45 putusan agar variasi kasus lebih luas.
3. Memperbaiki regex ekstraksi metadata berdasarkan pola putusan yang ditemukan.
4. Menggunakan model embedding seperti IndoBERT untuk menangkap kemiripan semantik.
5. Membuat ground-truth query manual agar evaluasi retrieval lebih valid.

## 10. Kesimpulan
Project ini menghasilkan sistem CBR berbasis Python yang dapat mengumpulkan putusan, membersihkan teks, mengekstrak metadata, membangun representasi kasus, melakukan retrieval kasus serupa, menggunakan solusi dari kasus lama, dan mengevaluasi hasil retrieval. Sistem ini sudah disusun dalam struktur project yang siap dijalankan dan diunggah ke GitHub sesuai ketentuan tugas.
