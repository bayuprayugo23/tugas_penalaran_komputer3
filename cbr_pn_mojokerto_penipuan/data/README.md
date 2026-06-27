# Folder Data

Folder ini akan terisi otomatis setelah pipeline dijalankan.

- `raw/` berisi teks putusan hasil unduh dan ekstraksi.
- `processed/` berisi `cases.csv` dan `cases.json` hasil representasi kasus.
- `eval/` berisi query evaluasi dan metrik.
- `results/` berisi hasil prediksi/reuse.

Jalankan:

```bash
python run_pipeline.py --limit 45
```
