from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_engineering import FeatureConfig, build_features
from src.explainability import (
    compute_shap_values,
    load_model as load_shap_model,
    plot_shap_values,
    prepare_shap_inputs,
    save_shap_values,
)
from src.modeling import ModelConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature engineering, imbalance-aware modeling, and SHAP explainability."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["features", "train", "explain", "all"],
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
        "--shap-out",
        default="models",
        help="Where to save SHAP artifacts.",
    )
    parser.add_argument(
        "--shap-plot-out",
        default="outputs/plots",
        help="Where to save SHAP plot images.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of background examples for SHAP.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of sample rows for SHAP analysis.",
    )
    parser.add_argument(
        "--shap-sample-index",
        type=int,
        default=0,
        help="Sample index for waterfall plot when saving SHAP plots.",
    )
    parser.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute SHAP values and save visualizations.",
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

    if args.stage in {"explain", "all"}:
        explain_artifacts = prepare_shap_inputs(
            Path(args.features_out),
            Path(args.shap_out),
            background_size=args.background_size,
            sample_size=args.sample_size,
        )
        print("SHAP explainability artifacts generated")
        print(f"SHAP artifacts: {Path(args.shap_out) / 'shap_background.joblib'}")
        print(f"SHAP sample CSV: {Path(args.shap_out) / 'shap_sample.csv'}")

        if args.compute_shap:
            shap_model = load_shap_model(Path(args.model_out))
            shap_values = compute_shap_values(
                shap_model,
                explain_artifacts["background"],
                explain_artifacts["sample"],
                explain_artifacts["feature_columns"],
            )
            save_shap_values(shap_values, Path(args.shap_out))
            plot_shap_values(shap_values, Path(args.shap_plot_out), sample_index=args.shap_sample_index)
            print(f"Saved SHAP values to {Path(args.shap_out) / 'shap_values.joblib'}")
            print(f"Saved SHAP plots to {Path(args.shap_plot_out)}")


if __name__ == "__main__":
    main()
