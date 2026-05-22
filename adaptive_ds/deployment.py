from __future__ import annotations

from pathlib import Path


SERVE_APP = """from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "model.pkl"
bundle = load(MODEL_PATH)
model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

app = FastAPI(title="AdaptiveDS Model Service", version="1.0.0")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, list[Any]]:
    data = pd.DataFrame(payload.rows)
    preds = model.predict(data)
    return {"predictions": preds.tolist()}
"""

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY serve_app.py /app/serve_app.py
COPY model.pkl /app/model.pkl
EXPOSE 8000
CMD ["uvicorn", "serve_app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE = """version: '3.9'
services:
  adaptive-ds-model:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
"""

REQS = """pandas>=2.0
joblib>=1.3
fastapi>=0.111
uvicorn>=0.30
scikit-learn>=1.3
"""


def generate_deployment_bundle(output_dir: Path, model_path: Path) -> dict[str, str]:
    deploy_dir = output_dir / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    serve_path = deploy_dir / "serve_app.py"
    dockerfile_path = deploy_dir / "Dockerfile"
    compose_path = deploy_dir / "docker-compose.yml"
    req_path = deploy_dir / "requirements.txt"
    model_copy_path = deploy_dir / "model.pkl"

    serve_path.write_text(SERVE_APP, encoding="utf-8")
    dockerfile_path.write_text(DOCKERFILE, encoding="utf-8")
    compose_path.write_text(DOCKER_COMPOSE, encoding="utf-8")
    req_path.write_text(REQS, encoding="utf-8")
    model_copy_path.write_bytes(model_path.read_bytes())

    return {
        "serve_app": str(serve_path.resolve()),
        "dockerfile": str(dockerfile_path.resolve()),
        "docker_compose": str(compose_path.resolve()),
        "serving_requirements": str(req_path.resolve()),
    }
