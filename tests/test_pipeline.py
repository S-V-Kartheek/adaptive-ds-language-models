from __future__ import annotations

from pathlib import Path

from adaptive_ds.pipeline import run_pipeline
from adaptive_ds.schemas import PipelineInput


def test_pipeline_generates_expected_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = PipelineInput(
        dataset_path=str(repo_root / "examples" / "customer_churn_sample.csv"),
        problem_statement="Predict churn risk for customers.",
        target_column="churn",
        user_expertise_level="beginner",
        random_state=7,
    )
    output = run_pipeline(config, str(tmp_path))
    assert output.task_type == "classification"
    assert output.selected_model is not None
    assert "f1_macro" in output.metrics
    assert Path(output.generated_files["dataset_profile_json"]).exists()
    assert Path(output.generated_files["task_recommendation_json"]).exists()
    assert Path(output.generated_files["model_pickle"]).exists()
