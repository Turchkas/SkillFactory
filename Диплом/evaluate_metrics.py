from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------
# Конфигурация
# -----------------------------
DATASET_NAME = "MilyaShams/nli_fever-ru_10k"
RETRIEVAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL_NAME = "cointegrated/rubert-base-cased-nli-threeway"

TOP_K = 5
RANDOM_SEED = 42
#MAX_EVAL_SAMPLES = 50
MAX_EVAL_SAMPLES = None  # например 300 для быстрого теста; None = весь test
CACHE_DIR = Path("cache_eval")
CACHE_DIR.mkdir(exist_ok=True)

# Для воспроизводимости
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------
# Вспомогательные функции
# -----------------------------
def simple_tokenize(text: str) -> list[str]:
    """
    Простая токенизация для BM25 baseline.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split() if t.strip()]
    return tokens


def label_to_hallucination(label: int) -> int:
    """
    Перевод 3-классовой разметки в бинарную:
    entailment (0) -> 0  (не галлюцинация)
    neutral (1) -> 1     (галлюцинация)
    contradiction (2) -> 1 (галлюцинация)
    """
    return 0 if label == 0 else 1


def normalize_label_name(label_name: str) -> str:
    """
    Нормализация имён меток модели.
    """
    return label_name.strip().lower()


# -----------------------------
# Загрузка датасета
# -----------------------------
def load_fever_ru():
    ds = load_dataset(DATASET_NAME)
    train_ds = ds["train"]
    test_ds = ds["test"]

    train_premises = train_ds["premise"]
    train_hypotheses = train_ds["hypothesis"]
    train_labels = train_ds["label"]

    test_premises = test_ds["premise"]
    test_hypotheses = test_ds["hypothesis"]
    test_labels = test_ds["label"]

    return (
        train_premises,
        train_hypotheses,
        train_labels,
        test_premises,
        test_hypotheses,
        test_labels,
    )


# -----------------------------
# BM25 baseline
# -----------------------------
class RuleBasedBM25Baseline:
    """
    Rule-based baseline:
    1) ищем top-k фрагментов через BM25;
    2) если в лучшем фрагменте есть все ключевые слова утверждения, считаем "подтверждено",
       иначе "галлюцинация".
    Это простой baseline, согласованный с описанием в главе 4.
    """

    def __init__(self, corpus_sentences: list[str]):
        self.corpus_sentences = corpus_sentences
        self.tokenized_corpus = [simple_tokenize(sent) for sent in corpus_sentences]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def predict(self, claim: str, top_k: int = TOP_K) -> int:
        claim_tokens = simple_tokenize(claim)
        scores = self.bm25.get_scores(claim_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        if len(top_indices) == 0:
            return 1  # галлюцинация

        best_text = self.corpus_sentences[int(top_indices[0])]
        best_tokens = set(simple_tokenize(best_text))

        # Простое правило: если все значимые токены claim встречаются, считаем подтверждением.
        # Для baseline этого достаточно.
        claim_token_set = set(claim_tokens)
        is_supported = claim_token_set.issubset(best_tokens)
        return 0 if is_supported else 1  # 0 = не галлюцинация, 1 = галлюцинация


# -----------------------------
# Dense retrieval: MiniLM + FAISS
# -----------------------------
class DenseRetriever:
    def __init__(self, corpus_sentences: list[str], model_name: str = RETRIEVAL_MODEL_NAME):
        self.corpus_sentences = corpus_sentences
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = None

    def build(self):
        embeddings = self.model.encode(
            self.corpus_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def retrieve(self, claim: str, top_k: int = TOP_K) -> list[str]:
        if self.index is None:
            raise RuntimeError("FAISS index is not built.")

        query_emb = self.model.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)
        result = []
        for idx in indices[0]:
            if idx != -1:
                result.append(self.corpus_sentences[int(idx)])
        return result


# -----------------------------
# NLI модель на основе RuBERT
# -----------------------------
class RuBERTNLI:
    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Сопоставим имена меток модели
        self.id2label = {
            int(k): normalize_label_name(v) for k, v in self.model.config.id2label.items()
        }

    def predict_label_name(self, claim: str, evidence: str) -> str:
        inputs = self.tokenizer(
            claim,
            evidence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()

        return self.id2label[pred_id]

    def predict_hallucination(self, claim: str, evidence: str) -> int:
        """
        Бинарное решение для одного evidence:
        entailment -> 0 (не галлюцинация)
        contradiction/neutral -> 1 (галлюцинация)
        """
        label_name = self.predict_label_name(claim, evidence)
        return 0 if label_name == "entailment" else 1


# -----------------------------
# Baseline: NLI без retrieval
# -----------------------------
class NLIWithoutRetrieval:
    def __init__(self, nli_model: RuBERTNLI, fixed_evidence: str):
        self.nli_model = nli_model
        self.fixed_evidence = fixed_evidence

    def predict(self, claim: str) -> int:
        return self.nli_model.predict_hallucination(claim, self.fixed_evidence)


# -----------------------------
# Предложенный метод: retrieval + NLI + агрегация по всем top-k
# -----------------------------
class ProposedMethod:
    def __init__(self, retriever: DenseRetriever, nli_model: RuBERTNLI):
        self.retriever = retriever
        self.nli_model = nli_model

    def predict(self, claim: str, top_k: int = TOP_K) -> int:
        evidences = self.retriever.retrieve(claim, top_k=top_k)

        if not evidences:
            return 1  # нет доказательств -> галлюцинация

        # ВАЖНО: согласовано с вашей главой 2.
        # Если хотя бы одно evidence = entailment, то не галлюцинация.
        for evidence in evidences:
            label_name = self.nli_model.predict_label_name(claim, evidence)
            if label_name == "entailment":
                return 0

        return 1


# -----------------------------
# Подсчёт метрик
# -----------------------------
def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


# -----------------------------
# Основной запуск
# -----------------------------
def main():
    print("Loading dataset...")
    (
        train_premises,
        train_hypotheses,
        train_labels,
        test_premises,
        test_hypotheses,
        test_labels,
    ) = load_fever_ru()

    # Корпус для retrieval: premises из train
    corpus_sentences = list(dict.fromkeys(train_premises))

    print(f"Train corpus size: {len(corpus_sentences)}")
    print(f"Test size: {len(test_labels)}")

    if MAX_EVAL_SAMPLES is not None:
        test_premises = test_premises[:MAX_EVAL_SAMPLES]
        test_hypotheses = test_hypotheses[:MAX_EVAL_SAMPLES]
        test_labels = test_labels[:MAX_EVAL_SAMPLES]

    print("Building dense retriever...")
    retriever = DenseRetriever(corpus_sentences)
    retriever.build()

    print("Loading NLI model...")
    nli_model = RuBERTNLI()

    # Для baseline NLI без retrieval берём фиксированный unrelated фрагмент
    fixed_evidence = corpus_sentences[0]

    rule_based = RuleBasedBM25Baseline(corpus_sentences)
    nli_no_retrieval = NLIWithoutRetrieval(nli_model, fixed_evidence)
    proposed = ProposedMethod(retriever, nli_model)

    y_true_bin = [label_to_hallucination(lbl) for lbl in test_labels]

    y_pred_rule = []
    y_pred_nli_no_retrieval = []
    y_pred_proposed = []

    print("Running evaluation...")
    for i, claim in enumerate(test_hypotheses):
        if i % 100 == 0:
            print(f"Processed {i}/{len(test_hypotheses)}")

        y_pred_rule.append(rule_based.predict(claim))
        y_pred_nli_no_retrieval.append(nli_no_retrieval.predict(claim))
        y_pred_proposed.append(proposed.predict(claim, top_k=TOP_K))

    rule_metrics = compute_metrics(y_true_bin, y_pred_rule)
    nli_no_retrieval_metrics = compute_metrics(y_true_bin, y_pred_nli_no_retrieval)
    proposed_metrics = compute_metrics(y_true_bin, y_pred_proposed)

    results = {
        "dataset": DATASET_NAME,
        "retrieval_model": RETRIEVAL_MODEL_NAME,
        "nli_model": NLI_MODEL_NAME,
        "top_k": TOP_K,
        "test_size": len(test_hypotheses),
        "rule_based": rule_metrics,
        "nli_without_retrieval": nli_no_retrieval_metrics,
        "proposed_method": proposed_metrics,
    }

    print("\n=== METRICS ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    out_path = CACHE_DIR / "metrics_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()