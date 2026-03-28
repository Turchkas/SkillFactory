from fastapi import FastAPI
from app.models.schemas import VerifyRequest, VerifyResponse, HealthResponse
from app.services.pipeline import HallucinationPipeline
from app.core.config import EMBEDDING_MODEL_NAME, NLI_MODEL_NAME, TOP_K

app = FastAPI(
    title="Semantic Hallucination Detector API",
    version="1.0.0",
    description="API for automatic detection of semantic hallucinations in Russian text",
)

pipeline = HallucinationPipeline()


@app.on_event("startup")
def startup_event() -> None:
    pipeline.initialize(force_rebuild_index=False)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if pipeline.ready else "initializing",
        device=pipeline.nli.device,
        embedding_model=EMBEDDING_MODEL_NAME,
        nli_model=NLI_MODEL_NAME,
        top_k=TOP_K,
    )


@app.post("/verify", response_model=VerifyResponse)
def verify_text(request: VerifyRequest) -> VerifyResponse:
    results = pipeline.analyze_text(request.text)
    return VerifyResponse(results=results)