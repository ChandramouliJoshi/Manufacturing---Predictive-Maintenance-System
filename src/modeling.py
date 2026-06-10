"""Imbalance-aware model training for predictive maintenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DEFAULT_TARGET = "machine_failure"
LEAKAGE_COLUMNS = {
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "failure_24h",
}


@dataclass(frozen=True)
class ModelConfig:
    data_path: Path = Path("data/processed/engineered_features.csv")
    model_path: Path = Path("models/production_failure_model.joblib")
    metrics_path: Path = Path("outputs/reports/imbalance_model_metrics.json")
    target: str = DEFAULT_TARGET
    model_name: str = "xgboost"
    test_size: float = 0.2
    min_precision: float = 0.8


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    drop_columns = {target, *LEAKAGE_COLUMNS}.intersection(df.columns)
    features = df.drop(columns=sorted(drop_columns))
    labels = df[target].astype(int)
    return features, labels


def class_imbalance_ratio(y: pd.Series) -> float:
    negative = int((y == 0).sum())
    positive = int((y == 1).sum())
    if positive == 0:
        raise ValueError("No positive failure examples found.")
    return negative / positive


def build_model(model_name: str, imbalance_ratio: float) -> Any:
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="aucpr",
                        scale_pos_weight=imbalance_ratio,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    raise ValueError(
        "Unknown model. Choose logistic_regression, random_forest, xgboost, or lightgbm."
    )


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, min_precision: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = []
    for idx, threshold in enumerate(thresholds):
        if precision[idx] >= min_precision:
            candidates.append((recall[idx], threshold))
    if not candidates:
        f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        return float(thresholds[int(np.argmax(f1_values))])
    return float(max(candidates, key=lambda item: item[0])[1])


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    min_precision: float,
) -> dict[str, Any]:
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities, min_precision)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_test, probabilities),
        "selected_threshold": threshold,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def train_model(config: ModelConfig) -> dict[str, Any]:
    df = load_training_data(config.data_path)
    x, y = split_features_target(df, config.target)
    imbalance_ratio = class_imbalance_ratio(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model(config.model_name, imbalance_ratio)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test, config.min_precision)
    metrics.update(
        {
            "model_name": config.model_name,
            "target": config.target,
            "imbalance_strategy": "class_weight_or_scale_pos_weight",
            "negative_to_positive_ratio": imbalance_ratio,
            "feature_columns": list(x.columns),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
        }
    )

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    config.metrics_path.write_text(
        json.dumps(make_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    return metrics
