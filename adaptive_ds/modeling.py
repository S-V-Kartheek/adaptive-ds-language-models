from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from .schemas import TaskType


@dataclass
class TrainResult:
    model_path: str | None
    selected_model: str | None
    metrics: dict[str, float]
    feature_ranking: list[tuple[str, float]]
    notes: list[str]


def _supervised_pipeline(estimator: BaseEstimator, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), num_cols),
            ("cat", Pipeline(categorical_steps), cat_cols),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _candidate_estimators(task: TaskType, random_state: int) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    if task == "classification":
        return {
            "LogisticRegression": (
                LogisticRegression(max_iter=2000, random_state=random_state),
                {"model__C": [0.5, 1.0, 2.0]},
            ),
            "RandomForestClassifier": (
                RandomForestClassifier(random_state=random_state),
                {"model__n_estimators": [100, 200], "model__max_depth": [None, 8]},
            ),
            "GradientBoostingClassifier": (
                GradientBoostingClassifier(random_state=random_state),
                {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
            ),
        }
    if task == "regression":
        return {
            "Ridge": (
                Ridge(),
                {"model__alpha": [0.5, 1.0, 2.0]},
            ),
            "RandomForestRegressor": (
                RandomForestRegressor(random_state=random_state),
                {"model__n_estimators": [100, 200], "model__max_depth": [None, 8]},
            ),
            "GradientBoostingRegressor": (
                GradientBoostingRegressor(random_state=random_state),
                {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
            ),
        }
    if task == "clustering":
        return {
            "KMeans": (KMeans(random_state=random_state, n_init=10), {"n_clusters": [2, 3, 4, 5]}),
            "DBSCAN": (DBSCAN(), {"eps": [0.3, 0.5, 0.8], "min_samples": [3, 5, 10]}),
            "GaussianMixture": (
                GaussianMixture(random_state=random_state),
                {"n_components": [2, 3, 4, 5]},
            ),
        }
    raise NotImplementedError(f"Task {task} is not supported in this implementation.")


def _evaluate_supervised(task: TaskType, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _feature_ranking_from_model(
    pipeline: Pipeline, feature_names: list[str], fallback_count: int = 8
) -> list[tuple[str, float]]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        values = np.asarray(model.feature_importances_, dtype=float)
        ranking = sorted(
            zip(transformed_names.tolist(), values.tolist()), key=lambda x: x[1], reverse=True
        )
        return [(name, float(score)) for name, score in ranking[:fallback_count]]
    if hasattr(model, "coef_"):
        transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        ranking = sorted(zip(transformed_names.tolist(), coef.tolist()), key=lambda x: x[1], reverse=True)
        return [(name, float(score)) for name, score in ranking[:fallback_count]]
    # Fallback to original columns if estimator has no accessible importances.
    return [(name, 0.0) for name in feature_names[:fallback_count]]


def train_and_select(
    task: TaskType,
    df: pd.DataFrame,
    target_column: str | None,
    output_model_path: str,
    random_state: int,
) -> TrainResult:
    notes: list[str] = []
    if task in {"classification", "regression"}:
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found for supervised task.")
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [col for col in X.columns if col not in num_cols]
        candidate_estimators = _candidate_estimators(task, random_state)

        stratify = y if task == "classification" and y.nunique() <= 30 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify
        )

        best_score = -np.inf
        best_pipe: Pipeline | None = None
        best_name: str | None = None
        best_metrics: dict[str, float] = {}

        for model_name, (estimator, param_grid) in candidate_estimators.items():
            pipe = _supervised_pipeline(estimator, num_cols, cat_cols)
            for params in ParameterGrid(param_grid):
                pipe.set_params(**params)
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_val)
                metrics = _evaluate_supervised(task, y_val, preds)
                score = metrics["f1_macro"] if task == "classification" else -metrics["rmse"]
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    best_name = model_name
                    best_metrics = metrics

        if best_pipe is None or best_name is None:
            raise RuntimeError("Model selection failed to produce a valid candidate.")

        dump(best_pipe, output_model_path)
        feature_ranking = _feature_ranking_from_model(best_pipe, X.columns.tolist())
        return TrainResult(
            model_path=output_model_path,
            selected_model=best_name,
            metrics={k: round(v, 6) for k, v in best_metrics.items()},
            feature_ranking=feature_ranking,
            notes=notes,
        )

    if task == "clustering":
        X = df.copy()
        X = X.select_dtypes(include=[np.number]).fillna(0.0)
        if X.shape[1] == 0:
            raise ValueError("Clustering requires at least one numeric feature.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        candidates = _candidate_estimators(task, random_state)
        best_score = -np.inf
        best_name = None
        best_model = None
        best_labels = None

        for name, (model, grid) in candidates.items():
            for params in ParameterGrid(grid):
                estimator = model.__class__(**{**model.get_params(), **params})
                if name == "GaussianMixture":
                    labels = estimator.fit_predict(X_scaled)
                else:
                    labels = estimator.fit_predict(X_scaled)
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2 or (name == "DBSCAN" and -1 in unique_labels and len(unique_labels) < 3):
                    continue
                mask = labels != -1
                if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                    continue
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = estimator
                    best_labels = labels

        if best_model is None or best_name is None:
            raise RuntimeError("No valid clustering model found for silhouette scoring.")

        dump({"scaler": scaler, "model": best_model}, output_model_path)
        metrics = {"silhouette_score": float(round(best_score, 6)), "n_clusters": int(len(np.unique(best_labels)))}
        ranking = [(col, 0.0) for col in X.columns[:8]]
        notes.append("Clustering feature ranking is placeholder because model-agnostic SHAP is not included in MVP.")
        return TrainResult(
            model_path=output_model_path,
            selected_model=best_name,
            metrics=metrics,
            feature_ranking=ranking,
            notes=notes,
        )

    raise NotImplementedError(f"Task '{task}' is not implemented yet.")
