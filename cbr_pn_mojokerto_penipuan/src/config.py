from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = DATA_DIR / "results"
LOG_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

DOMAIN = "Pidana Umum - Penipuan"
COURT = "PN MOJOKERTO"
TARGET_LIMIT = 45

# Halaman resmi Direktori Putusan MA RI untuk domain yang dipilih.
# Jika struktur URL MA berubah, URL ini bisa diganti tanpa mengubah script lain.
BASE_LIST_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/pengadilan/pn-mojokerto/kategori/penipuan-1.html"
BASE_DOMAIN_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/penipuan-1.html"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36; academic-cbr-project"
)

for path in [RAW_DIR, PROCESSED_DIR, EVAL_DIR, RESULTS_DIR, LOG_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
