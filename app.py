from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from adaptive_ds.pipeline import run_pipeline
from adaptive_ds.schemas import PipelineInput

app = FastAPI(title="AdaptiveDS-LM Orchestrator", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(payload: PipelineInput) -> dict:
    output_dir = Path("artifacts").resolve()
    result = run_pipeline(payload, str(output_dir))
    return result.model_dump()
