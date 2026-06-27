"""
Menjalankan pipeline CBR end-to-end:
1. Scraping/build case base
2. Case representation
3. Retrieval TF-IDF + SVM
4. Case/Solution reuse
5. Evaluation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=45, help="Jumlah putusan target")
    parser.add_argument("--max-pages", type=int, default=25, help="Maksimal halaman daftar untuk discan")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay antar-request scraping")
    parser.add_argument("--skip-scrape", action="store_true", help="Lewati scraping jika data/raw sudah tersedia")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_scrape:
        run([py, "01_scrape_cases.py", "--limit", str(args.limit), "--max-pages", str(args.max_pages), "--delay", str(args.delay)])
    run([py, "02_representation.py"])
    run([py, "03_retrieval.py"])
    run([py, "04_predict.py", "--k", str(args.k)])
    run([py, "05_evaluation.py", "--k", str(args.k)])
    print("\nPipeline selesai. Output utama berada di folder data/processed, data/eval, dan data/results.")


if __name__ == "__main__":
    main()
