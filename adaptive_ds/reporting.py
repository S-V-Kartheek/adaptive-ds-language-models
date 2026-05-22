from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .schemas import DatasetProfile, ModelRecommendation, PipelineOutput


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())


def write_dataset_profile(path: Path, profile: DatasetProfile) -> str:
    return write_json(path, profile.model_dump())


def write_model_recommendation(
    path: Path,
    task_type: str,
    confidence: float,
    recommendations: list[ModelRecommendation],
) -> str:
    payload = {
        "detected_task_type": task_type,
        "confidence_score": confidence,
        "top_models": [item.model_dump() for item in recommendations],
    }
    return write_json(path, payload)


def write_step_log(path: Path, lines: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path.resolve())


def build_eda_html(df: pd.DataFrame, profile: DatasetProfile, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = df.head(10).to_html(index=False)
    summary_df = df.describe(include="all").transpose().astype("object")
    summary_df = summary_df.where(summary_df.notna(), "")
    summary = summary_df.to_html()
    html = f"""
    <html>
      <head><title>AdaptiveDS Automated EDA Report</title></head>
      <body>
        <h1>AdaptiveDS Automated EDA Report</h1>
        <p>Rows: {profile.row_count} | Columns: {profile.column_count} | Missing Ratio: {profile.missing_ratio:.4f}</p>
        <h2>Preview</h2>
        {preview}
        <h2>Statistical Summary</h2>
        {summary}
      </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")
    return str(path.resolve())


def build_dashboard_html(
    df: pd.DataFrame,
    target_column: str | None,
    metrics: dict[str, float],
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = path.with_suffix(".png")
    plt.figure(figsize=(8, 4))
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        df[target_column].dropna().hist(bins=30)
        plt.title(f"Target Distribution: {target_column}")
    else:
        plt.text(0.5, 0.5, "No numeric target for distribution plot", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    metric_items = "".join([f"<li><b>{k}</b>: {v:.6f}</li>" for k, v in metrics.items()])
    html = f"""
    <html>
      <head><title>AdaptiveDS Performance Dashboard</title></head>
      <body>
        <h1>Performance Dashboard</h1>
        <ul>{metric_items}</ul>
        <img src="{plot_path.name}" alt="target distribution" />
      </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")
    return str(path.resolve())


def build_narrative(path: Path, output: PipelineOutput, problem_statement: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = output.notes[0] if output.notes else "No additional notes."
    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in output.metrics.items()]) or "No metrics generated."
    text = (
        "AdaptiveDS Insight Summary\n\n"
        f"Problem Statement: {problem_statement}\n"
        f"Detected Task: {output.task_type} (confidence={output.task_confidence:.2f})\n"
        f"Selected Model: {output.selected_model}\n"
        f"Validation Metrics: {metrics_str}\n\n"
        "Interpretation:\n"
        "- The selected model is recommended based on task-fit and validation performance.\n"
        "- Review the EDA report and recommendations JSON for model alternatives.\n"
        f"- Notes: {features}\n"
    )
    path.write_text(text, encoding="utf-8")
    return str(path.resolve())
