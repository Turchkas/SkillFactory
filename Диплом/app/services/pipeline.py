from typing import List, Dict, Any
from pathlib import Path

from app.core.config import (
    EMBEDDING_MODEL_NAME,
    NLI_MODEL_NAME,
    DATASET_NAME,
    TRAIN_SPLIT,
    TOP_K,
    CORPUS_SENTENCES_PATH,
    FAISS_INDEX_PATH,
)
from app.services.claim_extractor import ClaimExtractor
from app.services.dataset_loader import (
    load_fever_ru_split,
    prepare_unique_corpus,
    save_corpus,
    load_corpus,
)
from app.services.retriever import DenseRetriever
from app.services.nli_service import NLIService


class HallucinationPipeline:
    def __init__(self) -> None:
        self.claim_extractor = ClaimExtractor()
        self.retriever = DenseRetriever(EMBEDDING_MODEL_NAME)
        self.nli = NLIService(NLI_MODEL_NAME)
        self.ready = False

    def initialize(self, force_rebuild_index: bool = False) -> None:
        if (
            CORPUS_SENTENCES_PATH.exists()
            and FAISS_INDEX_PATH.exists()
            and not force_rebuild_index
        ):
            corpus_sentences = load_corpus(CORPUS_SENTENCES_PATH)
            self.retriever.load_index(FAISS_INDEX_PATH, corpus_sentences)
            self.ready = True
            return

        premises, _, _ = load_fever_ru_split(DATASET_NAME, TRAIN_SPLIT)
        corpus_sentences = prepare_unique_corpus(premises)

        save_corpus(corpus_sentences, CORPUS_SENTENCES_PATH)
        self.retriever.build_index(corpus_sentences)
        self.retriever.save_index(FAISS_INDEX_PATH)

        self.ready = True

    def analyze_text(self, text: str) -> List[Dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Pipeline is not initialized")

        claims = self.claim_extractor.extract_claims(text)
        results: List[Dict[str, Any]] = []

        for claim in claims:
            retrieved = self.retriever.retrieve(claim, TOP_K)
            evidence_texts = [item[0] for item in retrieved]
            evidence_scores = [item[1] for item in retrieved]

            predictions = self.nli.predict_many(claim, evidence_texts)
            hallucination = all(pred != "entailment" for pred in predictions)

            results.append(
                {
                    "claim": claim,
                    "evidences": [
                        {"text": ev, "score": score}
                        for ev, score in zip(evidence_texts, evidence_scores)
                    ],
                    "nli_predictions": predictions,
                    "hallucination": hallucination,
                }
            )

        return results