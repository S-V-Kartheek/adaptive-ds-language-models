from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal["regression", "classification", "clustering", "time_series"]
UserLevel = Literal["beginner", "intermediate", "expert"]


class PipelineInput(BaseModel):
    dataset_path: str = Field(..., description="Path to CSV/Parquet dataset")
    problem_statement: str = Field(..., min_length=8)
    target_column: str | None = None
    task_type: TaskType | None = None
    metric_preference: str | None = None
    user_expertise_level: UserLevel = "intermediate"
    max_search_trials: int = Field(default=10, ge=1, le=200)
    random_state: int = 42


class ModelRecommendation(BaseModel):
    model_name: str
    confidence: float
    rationale: str


class PipelineOutput(BaseModel):
    task_type: TaskType
    task_confidence: float
    recommended_models: list[ModelRecommendation]
    selected_model: str | None
    metrics: dict[str, float]
    generated_files: dict[str, str]
    notes: list[str]


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    missing_ratio: float
    numeric_columns: list[str]
    categorical_columns: list[str]
    column_stats: dict[str, dict[str, Any]]
    target_summary: dict[str, Any]
