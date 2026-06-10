from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_engineering import FeatureConfig, build_features
from src.modeling import ModelConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature engineering and imbalance-aware modeling."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["features", "train", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--data",
        default="data/raw/ai4i2020.csv",
        help="Path to the raw sensor CSV.",
    )
    parser.add_argument(
        "--features-out",
        default="data/processed/engineered_features.csv",
        help="Where to save engineered features.",
    )
    parser.add_argument(
        "--feature-metadata-out",
        default="models/feature_engineering.joblib",
        help="Where to save feature engineering metadata.",
    )
    parser.add_argument(
        "--target",
        default="machine_failure",
        help="Target column to predict.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Baseline or production classifier to train.",
    )
    parser.add_argument(
        "--model-out",
        default="models/production_failure_model.joblib",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--metrics-out",
        default="outputs/reports/imbalance_model_metrics.json",
        help="Where to save evaluation metrics.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.8,
        help="Minimum precision target for threshold selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_config = FeatureConfig(
        input_path=Path(args.data),
        output_path=Path(args.features_out),
        metadata_path=Path(args.feature_metadata_out),
    )

    if args.stage in {"features", "all"}:
        engineered = build_features(feature_config)
        print("Feature engineering complete")
        print(f"Features: {feature_config.output_path}")
        print(f"Feature metadata: {feature_config.metadata_path}")
        print(f"Rows: {len(engineered)}")

    if args.stage in {"train", "all"}:
        model_config = ModelConfig(
            data_path=Path(args.features_out),
            model_path=Path(args.model_out),
            metrics_path=Path(args.metrics_out),
            target=args.target,
            model_name=args.model,
            min_precision=args.min_precision,
        )

        metrics = train_model(model_config)
        print("Imbalance-aware modeling complete")
        print(f"Model: {model_config.model_path}")
        print(f"Metrics: {model_config.metrics_path}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Threshold: {metrics['selected_threshold']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
