from __future__ import annotations

from .schemas import DatasetProfile, ModelRecommendation, PipelineInput, TaskType

MODEL_CATALOG: dict[TaskType, list[str]] = {
    "regression": [
        "Ridge",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR",
        "MLPRegressor",
    ],
    "classification": [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
        "MLPClassifier",
    ],
    "clustering": ["KMeans", "DBSCAN", "GaussianMixture", "SpectralClustering"],
    "time_series": ["ARIMA", "Prophet", "RandomForestRegressor", "XGBoost"],
}

KEYWORD_HINTS: dict[TaskType, tuple[str, ...]] = {
    "classification": ("class", "churn", "fraud", "yes/no", "binary", "segment"),
    "regression": ("predict value", "estimate", "price", "amount", "revenue"),
    "clustering": ("cluster", "grouping", "segment customers", "unsupervised"),
    "time_series": ("forecast", "time series", "next month", "next quarter", "trend"),
}


def infer_task_type(inp: PipelineInput, profile: DatasetProfile) -> tuple[TaskType, float]:
    if inp.task_type:
        return inp.task_type, 0.98

    statement = inp.problem_statement.lower()
    hint_scores = {task: 0 for task in KEYWORD_HINTS}
    for task, keywords in KEYWORD_HINTS.items():
        for key in keywords:
            if key in statement:
                hint_scores[task] += 1

    best_keyword_task = max(hint_scores, key=hint_scores.get)
    if hint_scores[best_keyword_task] > 0:
        return best_keyword_task, min(0.9, 0.6 + 0.1 * hint_scores[best_keyword_task])

    if not inp.target_column or inp.target_column not in profile.column_stats:
        return "clustering", 0.75

    target_meta = profile.target_summary
    n_unique = int(target_meta.get("n_unique", 0))
    dtype = str(target_meta.get("dtype", "")).lower()
    if "int" in dtype or "float" in dtype:
        if n_unique <= 10:
            return "classification", 0.7
        return "regression", 0.78

    return "classification", 0.8


def recommend_models(task: TaskType) -> list[ModelRecommendation]:
    model_names = MODEL_CATALOG[task][:5]
    base_conf = 0.87
    recommendations: list[ModelRecommendation] = []
    for i, model_name in enumerate(model_names):
        confidence = max(0.55, base_conf - i * 0.08)
        recommendations.append(
            ModelRecommendation(
                model_name=model_name,
                confidence=round(confidence, 3),
                rationale=f"Selected as a strong {task} baseline for tabular data with robust generalization.",
            )
        )
    return recommendations
