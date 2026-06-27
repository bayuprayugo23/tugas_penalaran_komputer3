"""
Tahap 2 - Case Representation
Mengekstrak metadata, ringkasan fakta, pasal, amar putusan, label, dan fitur dasar dari raw text.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import COURT, DOMAIN, PROCESSED_DIR, RAW_DIR
from src.text_utils import excerpt_around, find_all_unique, first_regex, normalize_whitespace, safe_year, simple_tokenize


def extract_no_perkara(text: str, filename: str) -> str:
    patterns = [
        r"Nomor\s*[:\-]?\s*([0-9]+\s*/\s*Pid\.?B\s*/\s*20\d{2}\s*/\s*PN\s*\.?\s*(?:Mjk|Mojokerto))",
        r"([0-9]+\s*/\s*Pid\.?B\s*/\s*20\d{2}\s*/\s*PN\s*\.?\s*(?:Mjk|Mojokerto))",
        r"Nomor\s*[:\-]?\s*([^\n]{5,80}/PN\s*\.?\s*(?:Mjk|Mojokerto))",
    ]
    val = first_regex(patterns, text)
    return re.sub(r"\s+", "", val).replace("PN.Mjk", "PN Mjk") if val else filename


def extract_date(text: str, label: str = "putusan") -> str:
    # Mencari tanggal Indonesia; prioritas dekat kata putus/register/upload.
    keyword = {
        "putusan": r"(?:tanggal\s+putusan|diputus(?:kan)?\s+pada\s+hari|putusan)\D{0,80}",
        "register": r"(?:tanggal\s+register|register)\D{0,80}",
        "upload": r"(?:tanggal\s+upload|upload)\D{0,80}",
    }.get(label, "")
    month = r"Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember"
    patterns = []
    if keyword:
        patterns.append(keyword + rf"([0-3]?\d\s+(?:{month})\s+20\d{{2}})")
    patterns.append(rf"([0-3]?\d\s+(?:{month})\s+20\d{{2}})")
    return first_regex(patterns, text, flags=re.IGNORECASE | re.MULTILINE)


def extract_pasal(text: str) -> str:
    patterns = [
        r"Pasal\s+\d+[A-Za-z]?(?:\s+ayat\s*\(?\d+\)?)?(?:\s+ke-?\d+)?(?:\s+(?:KUHP|KUHAP))?",
        r"Pasal\s+\d+[A-Za-z]?\s*(?:jo\.?|juncto)\s*Pasal\s+\d+[A-Za-z]?",
    ]
    vals: list[str] = []
    for pat in patterns:
        vals.extend(find_all_unique(pat, text))
    # Domain penipuan umumnya Pasal 378 KUHP; tetap ambil semua pasal yang muncul.
    uniq = []
    for v in vals:
        if v.lower() not in [u.lower() for u in uniq]:
            uniq.append(v)
    return "; ".join(uniq[:12])


def extract_pihak(text: str) -> str:
    terdakwa = first_regex([
        r"Nama\s+lengkap\s*[:\-]?\s*([^\n;]{3,80})",
        r"Terdakwa\s*[:\-]?\s*([^\n;]{3,80})",
        r"terdakwa\s+([A-Z][A-Za-z\s.'-]{3,80})",
    ], text, flags=re.IGNORECASE | re.MULTILINE)
    jaksa = first_regex([
        r"Penuntut\s+Umum\s*[:\-]?\s*([^\n;]{3,100})",
        r"Jaksa\s+Penuntut\s+Umum\s*[:\-]?\s*([^\n;]{3,100})",
    ], text, flags=re.IGNORECASE | re.MULTILINE)
    parts = []
    if jaksa:
        parts.append(f"Penuntut Umum: {jaksa}")
    if terdakwa:
        parts.append(f"Terdakwa: {terdakwa}")
    return " | ".join(parts)


def extract_amar(text: str) -> str:
    # Bagian amar biasanya setelah MENGADILI; ambil sampai beberapa kata penutup.
    m = re.search(r"M\s*E\s*N\s*G\s*A\s*D\s*I\s*L\s*I|MENGADILI", text, flags=re.IGNORECASE)
    if not m:
        return excerpt_around(text, ["menyatakan terdakwa", "menjatuhkan pidana", "membebaskan terdakwa"], 1500)
    start = m.start()
    tail = text[start:]
    stop_patterns = [
        r"Demikianlah\s+diputuskan",
        r"Demikian\s+diputuskan",
        r"Hakim\s+Ketua",
        r"Panitera\s+Pengganti",
    ]
    stop = len(tail)
    for pat in stop_patterns:
        sm = re.search(pat, tail, flags=re.IGNORECASE)
        if sm:
            stop = min(stop, sm.start())
    return normalize_whitespace(tail[: min(stop, 2500)])


def classify_solution(amar: str, text: str) -> tuple[str, str]:
    src = f"{amar}\n{text}".lower()
    if "membebaskan terdakwa" in src or "tidak terbukti" in src:
        return "bebas", "Bebas/tidak terbukti"
    if "melepaskan terdakwa" in src:
        return "lepas", "Lepas dari segala tuntutan hukum"
    if "terbukti secara sah" in src or "menyatakan terdakwa terbukti" in src:
        label = "terbukti"
    else:
        label = "lainnya"

    # Ekstrak lama pidana penjara sederhana untuk kelas reuse.
    years = re.findall(r"(\d+)\s*(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)?\s*tahun", src)
    months = re.findall(r"(\d+)\s*bulan", src)
    max_months = 0
    if years:
        max_months = max(max_months, max(int(y) * 12 for y in years if y.isdigit()))
    if months:
        max_months = max(max_months, max(int(m) for m in months if m.isdigit()))
    if max_months == 0:
        return label, "Terbukti - pidana tidak terbaca"
    if max_months <= 12:
        sol = "Terbukti - pidana <= 1 tahun"
    elif max_months <= 24:
        sol = "Terbukti - pidana 1-2 tahun"
    else:
        sol = "Terbukti - pidana > 2 tahun"
    return label, sol


def represent_case(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    no_perkara = extract_no_perkara(text, path.stem)
    amar = extract_amar(text)
    label, solution_class = classify_solution(amar, text)
    facts = excerpt_around(text, ["dakwaan", "bahwa terdakwa", "menimbang", "barang bukti", "telah melakukan"], 1500)
    argument = excerpt_around(text, ["unsur", "pasal", "menimbang", "majelis hakim berpendapat"], 1300)
    tokens = simple_tokenize(text)
    return {
        "case_id": path.stem,
        "no_perkara": no_perkara,
        "pengadilan": COURT,
        "domain": DOMAIN,
        "tahun": safe_year(no_perkara) or safe_year(text),
        "tanggal_register": extract_date(text, "register"),
        "tanggal_putusan": extract_date(text, "putusan"),
        "tanggal_upload": extract_date(text, "upload"),
        "pasal": extract_pasal(text),
        "pihak": extract_pihak(text),
        "ringkasan_fakta": facts,
        "argumen_hukum": argument,
        "amar_putusan": amar,
        "label_putusan": label,
        "solution_class": solution_class,
        "word_count": len(tokens),
        "text_full": normalize_whitespace(text),
    }


def build_representation(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(p for p in raw_dir.glob("case_*.txt") if p.read_text(encoding="utf-8", errors="ignore").strip())
    if not files:
        raise FileNotFoundError("Tidak ada file data/raw/case_*.txt. Jalankan 01_scrape_cases.py terlebih dahulu.")
    rows = [represent_case(path) for path in files]
    df = pd.DataFrame(rows)

    manifest_path = PROCESSED_DIR / "manifest_downloads.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        cols = [c for c in ["case_id", "title", "detail_url", "pdf_url", "source_url", "status", "note"] if c in manifest.columns]
        df = df.merge(manifest[cols], on="case_id", how="left")
    else:
        df["detail_url"] = ""
        df["pdf_url"] = ""
        df["source_url"] = ""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "cases.csv"
    json_path = PROCESSED_DIR / "cases.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Representation selesai: {len(df)} kasus")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    build_representation(args.raw_dir)


if __name__ == "__main__":
    main()
