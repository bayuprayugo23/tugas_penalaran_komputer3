import re
from typing import Iterable

LEGAL_STOPWORDS = {
    "bahwa", "tersebut", "dengan", "dalam", "untuk", "pada", "yang", "dan", "atau",
    "ini", "itu", "oleh", "sebagai", "adalah", "para", "telah", "akan", "dari",
    "ke", "di", "sehingga", "karena", "maka", "serta", "yaitu", "yakni"
}

MONTHS_ID = (
    "januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember"
)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def clean_legal_text(text: str, lower: bool = False) -> str:
    """Membersihkan teks putusan tanpa menghilangkan konteks hukum penting."""
    if not isinstance(text, str):
        return ""
    patterns = [
        r"Direktori Putusan Mahkamah Agung Republik Indonesia",
        r"putusan\.mahkamahagung\.go\.id",
        r"Halaman\s+\d+\s+dari\s+\d+",
        r"Page\s+\d+\s+of\s+\d+",
        r"Disclaimer\s*:?[^\n]+",
        r"\bP U T U S A N\b",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = normalize_whitespace(text)
    return text.lower() if lower else text


def simple_tokenize(text: str) -> list[str]:
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ_/-]+", " ", text.lower())
    return [t for t in text.split() if len(t) > 2 and t not in LEGAL_STOPWORDS]


def excerpt_around(text: str, keywords: Iterable[str], max_chars: int = 1200) -> str:
    lower_text = text.lower()
    idx = -1
    for kw in keywords:
        idx = lower_text.find(kw.lower())
        if idx != -1:
            break
    if idx == -1:
        return normalize_whitespace(text[:max_chars])
    start = max(0, idx - 250)
    end = min(len(text), idx + max_chars)
    return normalize_whitespace(text[start:end])


def first_regex(patterns: Iterable[str], text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            return normalize_whitespace(match.group(1))
    return ""


def find_all_unique(pattern: str, text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> list[str]:
    vals = []
    for match in re.finditer(pattern, text, flags=flags):
        item = normalize_whitespace(match.group(0))
        if item and item.lower() not in [v.lower() for v in vals]:
            vals.append(item)
    return vals


def safe_year(text: str) -> str:
    m = re.search(r"/(20\d{2})/", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(20\d{2})\b", text)
    return m.group(1) if m else ""
