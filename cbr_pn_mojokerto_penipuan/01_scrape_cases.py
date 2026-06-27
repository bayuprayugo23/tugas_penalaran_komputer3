"""
Tahap 1 - Membangun Case Base
Scraping 45 putusan PN Mojokerto kategori Pidana Umum - Penipuan dari Direktori Putusan MA RI.

Catatan etis:
- Gunakan untuk tugas akademik.
- Script memakai delay agar tidak membebani server.
- Apabila website MA sedang lambat, jalankan ulang atau naikkan --delay.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from tqdm import tqdm

from src.config import BASE_LIST_URL, COURT, DOMAIN, LOG_DIR, RAW_DIR, USER_AGENT
from src.text_utils import clean_legal_text, normalize_whitespace


@dataclass
class CaseDownload:
    case_id: str
    title: str
    detail_url: str
    pdf_url: str
    source_url: str
    raw_path: str
    status: str
    char_count: int
    note: str = ""


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "id-ID,id;q=0.9,en;q=0.8"})
    return session


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scrape_cases")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "cleaning.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def make_page_url(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    # Struktur umum Direktori MA: .../kategori/penipuan-1/page/2.html
    if base_url.endswith(".html"):
        return base_url[:-5] + f"/page/{page}.html"
    return base_url.rstrip("/") + f"/page/{page}.html"


def fetch(session: requests.Session, url: str, timeout: int = 35, retries: int = 3, sleep: float = 1.5) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(sleep * attempt)
    raise RuntimeError(f"Gagal mengambil URL {url}: {last_error}")


def extract_case_links(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        text = normalize_whitespace(a.get_text(" "))
        haystack = f"{text} {href}".lower()
        is_detail = "/direktori/putusan/" in href or "/putusan/" in href
        is_relevant = "pn mojokerto" in haystack or "pn mjk" in haystack or "mojokerto" in haystack
        if is_detail and is_relevant and href not in seen:
            seen.add(href)
            found.append((text or href, href))
    return found


def discover_case_urls(limit: int, max_pages: int, delay: float, list_url: str = BASE_LIST_URL) -> list[tuple[str, str, str]]:
    session = make_session()
    case_links: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        page_url = make_page_url(list_url, page)
        resp = fetch(session, page_url, sleep=delay)
        links = extract_case_links(resp.text, page_url)
        for title, detail_url in links:
            if detail_url not in seen:
                case_links.append((title, detail_url, page_url))
                seen.add(detail_url)
            if len(case_links) >= limit:
                return case_links
        time.sleep(delay)
    return case_links


def find_pdf_url(detail_html: str, detail_url: str) -> str:
    soup = BeautifulSoup(detail_html, "lxml")
    candidates: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(detail_url, a["href"])
        text = (a.get_text(" ") or "").lower()
        haystack = f"{text} {href}".lower()
        if href.lower().endswith(".pdf") or "download" in haystack or "/pdf/" in haystack or "unduh" in haystack:
            candidates.append(href)
    # Prioritaskan link yang benar-benar PDF/download.
    for c in candidates:
        if c.lower().endswith(".pdf") or "/pdf/" in c.lower() or "download" in c.lower():
            return c
    return candidates[0] if candidates else ""


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return clean_legal_text(text)


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    with io.BytesIO(pdf_bytes) as bio:
        return clean_legal_text(extract_text(bio) or "")


def extract_text_from_detail(session: requests.Session, detail_url: str, delay: float) -> tuple[str, str, str]:
    detail_resp = fetch(session, detail_url, sleep=delay)
    detail_html = detail_resp.text
    pdf_url = find_pdf_url(detail_html, detail_url)
    if pdf_url:
        try:
            pdf_resp = fetch(session, pdf_url, timeout=60, sleep=delay)
            ctype = pdf_resp.headers.get("content-type", "").lower()
            if "pdf" in ctype or pdf_url.lower().endswith(".pdf"):
                text = pdf_bytes_to_text(pdf_resp.content)
                if len(text) > 500:
                    return text, pdf_url, "pdf"
        except Exception:
            # Jika PDF gagal, fallback ke HTML detail.
            pass
    return html_to_text(detail_html), pdf_url, "html"


def write_manifest(rows: Iterable[CaseDownload]) -> Path:
    path = RAW_DIR.parent / "processed" / "manifest_downloads.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [
            "case_id", "title", "detail_url", "pdf_url", "source_url", "raw_path", "status", "char_count", "note"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def scrape(limit: int = 45, max_pages: int = 15, delay: float = 1.5, list_url: str = BASE_LIST_URL) -> list[CaseDownload]:
    logger = setup_logger()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Mulai scraping domain=%s court=%s target=%s", DOMAIN, COURT, limit)
    logger.info("URL daftar: %s", list_url)

    links = discover_case_urls(limit=limit, max_pages=max_pages, delay=delay, list_url=list_url)
    logger.info("Ditemukan %s link kandidat putusan", len(links))
    if len(links) < limit:
        logger.warning("Jumlah link kandidat kurang dari target. Coba naikkan --max-pages atau cek URL filter Direktori MA.")

    session = make_session()
    rows: list[CaseDownload] = []
    for idx, (title, detail_url, source_url) in enumerate(tqdm(links[:limit], desc="Download putusan"), start=1):
        case_id = f"case_{idx:03d}"
        raw_path = RAW_DIR / f"{case_id}.txt"
        status = "ok"
        note = ""
        pdf_url = ""
        text = ""
        try:
            text, pdf_url, source_type = extract_text_from_detail(session, detail_url, delay=delay)
            text = clean_legal_text(text)
            if len(text) < 1000:
                status = "warning"
                note = f"Teks pendek ({len(text)} karakter), sumber={source_type}; perlu cek manual."
            raw_path.write_text(text, encoding="utf-8")
            logger.info("%s tersimpan: %s karakter | %s", case_id, len(text), detail_url)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            note = str(exc)
            raw_path.write_text("", encoding="utf-8")
            logger.exception("Gagal memproses %s: %s", detail_url, exc)
        rows.append(CaseDownload(
            case_id=case_id,
            title=title,
            detail_url=detail_url,
            pdf_url=pdf_url,
            source_url=source_url,
            raw_path=str(raw_path.relative_to(RAW_DIR.parent.parent)),
            status=status,
            char_count=len(text),
            note=note,
        ))
        time.sleep(delay)

    manifest = write_manifest(rows)
    logger.info("Manifest tersimpan: %s", manifest)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=45, help="Jumlah putusan yang diunduh")
    parser.add_argument("--max-pages", type=int, default=20, help="Maksimal halaman daftar yang discan")
    parser.add_argument("--delay", type=float, default=1.5, help="Jeda antar-request dalam detik")
    parser.add_argument("--list-url", type=str, default=BASE_LIST_URL, help="URL daftar putusan filter PN Mojokerto + Penipuan")
    args = parser.parse_args()
    rows = scrape(limit=args.limit, max_pages=args.max_pages, delay=args.delay, list_url=args.list_url)
    ok = sum(1 for r in rows if r.status in {"ok", "warning"})
    print(f"Selesai scraping. Berhasil: {ok}/{args.limit}. Lihat data/raw dan data/processed/manifest_downloads.csv")


if __name__ == "__main__":
    main()
