from datasets import load_dataset
from typing import List, Tuple
from pathlib import Path


def load_fever_ru_split(dataset_name: str, split: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Returns:
        premises: evidence sentences
        hypotheses: claims
        labels: 0/1/2 labels from dataset
    """
    ds = load_dataset(dataset_name, split=split)
    premises = ds["premise"]
    hypotheses = ds["hypothesis"]
    labels = ds["label"]
    return premises, hypotheses, labels


def prepare_unique_corpus(premises: List[str]) -> List[str]:
    unique_sentences = list(dict.fromkeys([p.strip() for p in premises if p and p.strip()]))
    return unique_sentences


def save_corpus(sentences: List[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent.replace("\n", " ").strip() + "\n")


def load_corpus(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]