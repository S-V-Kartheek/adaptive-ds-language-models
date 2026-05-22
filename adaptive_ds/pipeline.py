from __future__ import annotations

from pathlib import Path

from .deployment import generate_deployment_bundle
from .io_utils import load_dataset
from .modeling import train_and_select
from .profiler import build_dataset_profile
from .reporting import (
    build_dashboard_html,
    build_eda_html,
    build_narrative,
    write_dataset_profile,
    write_model_recommendation,
    write_step_log,
)
from .schemas import PipelineInput, PipelineOutput
from .task_router import infer_task_type, recommend_models


def run_pipeline(config: PipelineInput, output_dir: str) -> PipelineOutput:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    step_log_lines = []
    step_log_lines.append("L1: Ingestion started")
    df = load_dataset(config.dataset_path)
    profile = build_dataset_profile(df, config.target_column)
    step_log_lines.append("L1: Dataset profiling completed")

    profile_path = write_dataset_profile(out_dir / "dataset_profile.json", profile)

    step_log_lines.append("L2: Context understanding started")
    task_type, confidence = infer_task_type(config, profile)
    recommendations = recommend_models(task_type)
    recommendation_path = write_model_recommendation(
        out_dir / "task_recommendation.json", task_type, confidence, recommendations
    )
    step_log_lines.append("L2/L3: Task routing and model recommendation completed")

    model_path = out_dir / "model.pkl"
    train_result = train_and_select(
        task=task_type,
        df=df,
        target_column=config.target_column,
        output_model_path=str(model_path),
        random_state=config.random_state,
    )
    step_log_lines.append("L4: AutoML model search completed")

    notes = list(train_result.notes)
    if train_result.feature_ranking:
        top_features = ", ".join([f"{name} ({score:.3f})" for name, score in train_result.feature_ranking[:5]])
        notes.append(f"Top feature signals: {top_features}")

    output = PipelineOutput(
        task_type=task_type,
        task_confidence=round(confidence, 4),
        recommended_models=recommendations,
        selected_model=train_result.selected_model,
        metrics=train_result.metrics,
        generated_files={},
        notes=notes,
    )

    eda_report_path = build_eda_html(df, profile, out_dir / "eda_report.html")
    dashboard_path = build_dashboard_html(df, config.target_column, output.metrics, out_dir / "dashboard.html")
    step_log_lines.append("L5: Explanation and reporting artifacts generated")

    output.generated_files.update(
        {
            "dataset_profile_json": profile_path,
            "task_recommendation_json": recommendation_path,
            "model_pickle": str(model_path.resolve()),
            "eda_report_html": eda_report_path,
            "performance_dashboard_html": dashboard_path,
        }
    )

    insight_path = build_narrative(out_dir / "insight_report.txt", output, config.problem_statement)
    output.generated_files["insight_report_txt"] = insight_path

    deployment_files = generate_deployment_bundle(out_dir, model_path)
    output.generated_files.update(deployment_files)

    step_log_lines.append("L6: Deployment bundle generated")
    step_log_lines.append("L6: Pipeline completed")
    steps_path = write_step_log(out_dir / "pipeline_steps.log", step_log_lines)
    output.generated_files["pipeline_step_log"] = steps_path
    return output
