# AdaptiveDS Language Models

This repository implements an end-to-end **AdaptiveDS-LM** reference pipeline based on the proposal document:

- L1: ingestion + dataset profiling
- L2/L3: intent understanding + task routing + model recommendation
- L4: AutoML-style model search and training
- L5: explainability-oriented reporting outputs
- L6: deployable serving bundle generation

## What This Implementation Produces

For each run, the pipeline generates:

1. `dataset_profile.json` (machine-readable schema/profile)
2. `task_recommendation.json` (task type + confidence + top models)
3. `model.pkl` (trained model artifact)
4. `eda_report.html` (automated EDA report)
5. `dashboard.html` (performance dashboard)
6. `insight_report.txt` (natural language summary)
7. `deployment/` bundle with `serve_app.py`, `Dockerfile`, `docker-compose.yml`
8. `pipeline_steps.log` (middle-step trace from L1 to L6)

## Inputs

The pipeline accepts a JSON config:

```json
{
  "dataset_path": "examples/customer_churn_sample.csv",
  "problem_statement": "Predict customer churn within 30 days and explain major factors.",
  "target_column": "churn",
  "task_type": null,
  "metric_preference": null,
  "user_expertise_level": "beginner",
  "max_search_trials": 12,
  "random_state": 42
}
```

`dataset_path` supports CSV and Parquet.

## Run

```bash
pip install -r requirements.txt
python -m adaptive_ds.cli --config examples/run_config.json --output-dir artifacts
```

## FastAPI Orchestrator

```bash
uvicorn app:app --reload --port 8000
```

POST `/run` with the same JSON schema to execute the pipeline.

## Tests

```bash
pytest -q
```
