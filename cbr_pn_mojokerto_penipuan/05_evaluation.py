"""
Tahap 5 - Model Evaluation
Menghitung metrik retrieval dan prediksi solusi: Accuracy/Hit@K, Precision@K, Recall@K, F1@K,
serta classification metrics untuk predicted solution jika ground truth tersedia.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EVAL_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.text_utils import clean_legal_text


def load_cases() -> pd.DataFrame:
    path = PROCESSED_DIR / "cases.csv"
    if not path.exists():
        raise FileNotFoundError("data/processed/cases.csv belum ada.")
    return pd.read_csv(path).fillna("")


def retrieve_ids(query: str, k: int, df: pd.DataFrame) -> list[str]:
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(MODELS_DIR / "tfidf_matrix.joblib")
    q_vec = vectorizer.transform([clean_legal_text(query, lower=True)])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    return df.iloc[top_idx]["case_id"].astype(str).tolist()


def eval_retrieval(queries: list[dict], k: int = 5) -> pd.DataFrame:
    df = load_cases()
    rows = []
    for q in queries:
        top_ids = retrieve_ids(q["query"], k=k, df=df)
        gt = str(q.get("ground_truth_case_id", ""))
        hit = int(gt in top_ids)
        precision = hit / max(k, 1)
        recall = hit  # satu ground-truth relevan per query
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({
            "query_id": q.get("query_id", ""),
            "ground_truth_case_id": gt,
            "top_k_case_ids": ", ".join(top_ids),
            "hit_at_k": hit,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1,
        })
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "query_id": "SUMMARY",
        "ground_truth_case_id": "",
        "top_k_case_ids": "",
        "hit_at_k": detail["hit_at_k"].mean() if not detail.empty else 0,
        "precision_at_k": detail["precision_at_k"].mean() if not detail.empty else 0,
        "recall_at_k": detail["recall_at_k"].mean() if not detail.empty else 0,
        "f1_at_k": detail["f1_at_k"].mean() if not detail.empty else 0,
    }])
    return pd.concat([detail, summary], ignore_index=True)


def eval_prediction() -> pd.DataFrame:
    pred_path = RESULTS_DIR / "predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame([{"note": "predictions.csv belum tersedia. Jalankan 04_predict.py terlebih dahulu."}])
    df = pd.read_csv(pred_path).fillna("")
    valid = df[(df["ground_truth_solution"].astype(str) != "") & (df["predicted_solution"].astype(str) != "")]
    if valid.empty:
        return pd.DataFrame([{"note": "Tidak ada ground_truth_solution valid untuk evaluasi prediksi."}])
    y_true = valid["ground_truth_solution"].astype(str)
    y_pred = valid["predicted_solution"].astype(str)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out = pd.DataFrame(report).transpose()
    out.loc["accuracy", "precision"] = accuracy_score(y_true, y_pred)
    return out


def plot_retrieval_metrics(metrics: pd.DataFrame) -> None:
    summary = metrics[metrics["query_id"] == "SUMMARY"]
    if summary.empty:
        return
    vals = summary[["hit_at_k", "precision_at_k", "recall_at_k", "f1_at_k"]].iloc[0]
    ax = vals.plot(kind="bar")
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval Performance TF-IDF Cosine Similarity")
    ax.set_ylabel("Score")
    plt.tight_layout()
    out = EVAL_DIR / "retrieval_metrics_chart.png"
    plt.savefig(out, dpi=200)
    plt.close()


def evaluate(queries_path: Path = EVAL_DIR / "queries.json", k: int = 5) -> None:
    if not queries_path.exists():
        raise FileNotFoundError("data/eval/queries.json belum ada. Jalankan 03_retrieval.py dulu.")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_metrics = eval_retrieval(queries, k=k)
    retrieval_path = EVAL_DIR / "retrieval_metrics.csv"
    retrieval_metrics.to_csv(retrieval_path, index=False, encoding="utf-8-sig")
    pred_metrics = eval_prediction()
    pred_path = EVAL_DIR / "prediction_metrics.csv"
    pred_metrics.to_csv(pred_path, encoding="utf-8-sig")
    plot_retrieval_metrics(retrieval_metrics)
    print(f"Retrieval metrics: {retrieval_path}")
    print(f"Prediction metrics: {pred_path}")
    print("Ringkasan Retrieval:")
    print(retrieval_metrics.tail(1).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=EVAL_DIR / "queries.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.queries, k=args.k)


if __name__ == "__main__":
    main()
