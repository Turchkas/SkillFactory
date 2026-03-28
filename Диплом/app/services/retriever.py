from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
from pathlib import Path


class DenseRetriever:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus_sentences: List[str] = []

    def build_index(self, corpus_sentences: List[str]) -> None:
        self.corpus_sentences = corpus_sentences
        embeddings = self.model.encode(
            corpus_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index

    def save_index(self, index_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Index is not built")
        faiss.write_index(self.index, str(index_path))

    def load_index(self, index_path: Path, corpus_sentences: List[str]) -> None:
        self.index = faiss.read_index(str(index_path))
        self.corpus_sentences = corpus_sentences

    def retrieve(self, claim: str, top_k: int) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index is not initialized")

        claim_embedding = self.model.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(claim_embedding, top_k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.corpus_sentences[idx], float(score)))
        return results