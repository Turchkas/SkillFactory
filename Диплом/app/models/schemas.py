from pydantic import BaseModel, Field
from typing import List


class VerifyRequest(BaseModel):
    text: str = Field(..., description="Input text for hallucination analysis")


class EvidenceItem(BaseModel):
    text: str
    score: float


class ClaimResult(BaseModel):
    claim: str
    evidences: List[EvidenceItem]
    nli_predictions: List[str]
    hallucination: bool


class VerifyResponse(BaseModel):
    results: List[ClaimResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    embedding_model: str
    nli_model: str
    top_k: int
