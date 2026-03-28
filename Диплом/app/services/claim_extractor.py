from natasha import Segmenter, Doc
from typing import List


class ClaimExtractor:
    def __init__(self) -> None:
        self.segmenter = Segmenter()

    def extract_claims(self, text: str) -> List[str]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        claims = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return claims