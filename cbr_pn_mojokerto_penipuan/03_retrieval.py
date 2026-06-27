"""
Tahap 3 - Case Retrieval
Membangun index TF-IDF, retrieval top-k dengan cosine similarity, dan model SVM sederhana jika label tersedia.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.config import EVAL_DIR, MODELS_DIR, PROCESSED_DIR
from src.text_utils import clean_legal_text


RETRIEVAL_COLUMNS = ["ringkasan_fakta", "argumen_hukum", "pasal", "amar_putusan", "solution_class"]


def load_cases(cases_path: Path = PROCESSED_DIR / "cases.csv") -> pd.DataFrame:
    if not cases_path.exists():
        raise FileNotFoundError("File data/processed/cases.csv belum ada. Jalankan 02_representation.py dulu.")
    df = pd.read_csv(cases_path).fillna("")
    if df.empty:
        raise ValueError("cases.csv kosong.")
    return df


def make_document_text(df: pd.DataFrame) -> pd.Series:
    text = df[RETRIEVAL_COLUMNS].astype(str).agg(" ".join, axis=1)
    return text.map(lambda x: clean_legal_text(x, lower=True))


def build_tfidf_index(df: pd.DataFrame):
    docs = make_document_text(df)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=25000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, MODELS_DIR / "tfidf_matrix.joblib")
    df[["case_id", "no_perkara", "solution_class", "label_putusan"]].to_csv(MODELS_DIR / "case_index.csv", index=False)
    return vectorizer, matrix


def retrieve(query: str, k: int = 5, df: pd.DataFrame | None = None, vectorizer=None, matrix=None) -> pd.DataFrame:
    if df is None:
        df = load_cases()
    if vectorizer is None:
        vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    if matrix is None:
        matrix = joblib.load(MODELS_DIR / "tfidf_matrix.joblib")
    q_vec = vectorizer.transform([clean_legal_text(query, lower=True)])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    result = df.iloc[top_idx].copy()
    result.insert(0, "similarity", scores[top_idx])
    return result[["similarity", "case_id", "no_perkara", "pasal", "solution_class", "amar_putusan"]]


def train_svm_if_possible(df: pd.DataFrame, vectorizer, matrix) -> None:
    labels = df["label_putusan"].fillna("lainnya").astype(str)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if labels.nunique() < 2 or len(df) < 10:
        pd.DataFrame([{
            "note": "SVM tidak dilatih karena label hanya satu kelas atau jumlah data terlalu sedikit.",
            "jumlah_data": len(df),
            "jumlah_label": labels.nunique(),
        }]).to_csv(EVAL_DIR / "classification_report.csv", index=False)
        return
    try:
        stratify = labels if labels.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            matrix, labels, test_size=0.2, random_state=42, stratify=stratify
        )
        clf = LinearSVC(random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(EVAL_DIR / "classification_report.csv")
        joblib.dump(clf, MODELS_DIR / "svm_label_model.joblib")
    except Exception as exc:  # noqa: BLE001
        pd.DataFrame([{"note": f"SVM gagal dilatih: {exc}"}]).to_csv(EVAL_DIR / "classification_report.csv", index=False)


def generate_eval_queries(df: pd.DataFrame, n: int = 10) -> Path:
    # Query evaluasi otomatis: potongan ringkasan fakta dari beberapa kasus sebagai pseudo-query.
    sample = df.sample(min(n, len(df)), random_state=42) if len(df) > n else df
    queries = []
    for _, row in sample.iterrows():
        text = str(row.get("ringkasan_fakta", ""))
        query = text[:700] if len(text) > 80 else " ".join([
            str(row.get("pasal", "")), str(row.get("argumen_hukum", ""))[:500]
        ])
        queries.append({
            "query_id": f"q_{len(queries)+1:03d}",
            "query": query,
            "ground_truth_case_id": row["case_id"],
            "ground_truth_solution": row.get("solution_class", ""),
        })
    path = EVAL_DIR / "queries.json"
    path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_retrieval(cases_path: Path = PROCESSED_DIR / "cases.csv") -> None:
    df = load_cases(cases_path)
    vectorizer, matrix = build_tfidf_index(df)
    train_svm_if_possible(df, vectorizer, matrix)
    q_path = generate_eval_queries(df)
    print(f"TF-IDF index selesai untuk {len(df)} kasus.")
    print(f"Model tersimpan di: {MODELS_DIR}")
    print(f"Query evaluasi: {q_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=PROCESSED_DIR / "cases.csv")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.query:
        df = load_cases(args.cases)
        if not (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
            build_retrieval(args.cases)
        print(retrieve(args.query, args.k, df=df).to_string(index=False))
    else:
        build_retrieval(args.cases)


if __name__ == "__main__":
    main()
