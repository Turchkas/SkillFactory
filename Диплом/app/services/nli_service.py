from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List


class NLIService:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def predict_label_id(self, claim: str, evidence: str) -> int:
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
            prediction = torch.argmax(outputs.logits, dim=1).item()
        return prediction

    @staticmethod
    def id_to_label(pred_id: int) -> str:
        mapping = {
            0: "entailment",
            1: "contradiction",
            2: "neutral",
        }
        return mapping.get(pred_id, "unknown")

    def predict_label(self, claim: str, evidence: str) -> str:
        pred_id = self.predict_label_id(claim, evidence)
        return self.id_to_label(pred_id)

    def predict_many(self, claim: str, evidences: List[str]) -> List[str]:
        return [self.predict_label(claim, evidence) for evidence in evidences]
