"""
Tahap 4 - Case/Solution Reuse
Mengambil top-k kasus termirip dan menghasilkan prediksi solusi dengan weighted similarity vote.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EVAL_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.text_utils import clean_legal_text, normalize_whitespace


def load_cases(cases_path: Path = PROCESSED_DIR / "cases.csv") -> pd.DataFrame:
    if not cases_path.exists():
        raise FileNotFoundError("cases.csv belum ada. Jalankan tahap 02 dan 03 terlebih dahulu.")
    return pd.read_csv(cases_path).fillna("")


def retrieve_local(query: str, k: int = 5, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_cases()
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(MODELS_DIR / "tfidf_matrix.joblib")
    q_vec = vectorizer.transform([clean_legal_text(query, lower=True)])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    result = df.iloc[top_idx].copy()
    result.insert(0, "similarity", scores[top_idx])
    return result


def weighted_vote(top_k: pd.DataFrame) -> str:
    weights: dict[str, float] = defaultdict(float)
    for _, row in top_k.iterrows():
        sol = str(row.get("solution_class", "Terbukti - pidana tidak terbaca"))
        weights[sol] += float(row.get("similarity", 0.0))
    if not weights:
        return "Tidak ada solusi"
    return max(weights.items(), key=lambda item: item[1])[0]


def summarize_solution(top_k: pd.DataFrame, predicted_class: str) -> str:
    chosen = top_k[top_k["solution_class"].astype(str) == predicted_class]
    if chosen.empty:
        chosen = top_k
    best = chosen.iloc[0]
    amar = normalize_whitespace(str(best.get("amar_putusan", "")))
    if len(amar) > 900:
        amar = amar[:900] + "..."
    return f"Prediksi: {predicted_class}. Dasar reuse dari kasus paling mirip {best.get('no_perkara', best.get('case_id'))}: {amar}"


def predict_outcome(query: str, k: int = 5, df: pd.DataFrame | None = None) -> dict:
    top_k = retrieve_local(query, k=k, df=df)
    pred = weighted_vote(top_k)
    return {
        "predicted_solution": pred,
        "solution_text": summarize_solution(top_k, pred),
        "top_5_case_ids": ", ".join(top_k["case_id"].astype(str).tolist()),
        "top_5_scores": ", ".join(f"{s:.4f}" for s in top_k["similarity"].tolist()),
    }


def batch_predict(queries_path: Path = EVAL_DIR / "queries.json", k: int = 5) -> pd.DataFrame:
    if not queries_path.exists():
        raise FileNotFoundError("data/eval/queries.json belum ada. Jalankan 03_retrieval.py dulu.")
    df = load_cases()
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    rows = []
    for q in queries:
        pred = predict_outcome(q["query"], k=k, df=df)
        rows.append({
            "query_id": q.get("query_id", ""),
            "query": q.get("query", ""),
            "ground_truth_case_id": q.get("ground_truth_case_id", ""),
            "ground_truth_solution": q.get("ground_truth_solution", ""),
            **pred,
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "predictions.csv"
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Prediksi selesai: {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--queries", type=Path, default=EVAL_DIR / "queries.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.query:
        print(predict_outcome(args.query, k=args.k)["solution_text"])
    else:
        batch_predict(args.queries, k=args.k)


if __name__ == "__main__":
    main()
