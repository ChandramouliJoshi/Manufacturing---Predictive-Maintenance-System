"""Prepare SHAP inputs and save background/sample artifacts for explainability."""

from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import pandas as pd


def prepare_shap_inputs(
    data_path: Path,
    output_dir: Path,
    background_size: int = 100,
    sample_size: int = 20,
    random_state: int = 42,
):
    df = pd.read_csv(data_path)

    # Heuristic: drop known leakage/target columns if present
    drop_cols = [
        "machine_failure",
        "failure_24h",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    features = df[feature_cols].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    background = features.sample(n=min(background_size, len(features)), random_state=random_state)
    sample = features.sample(n=min(sample_size, len(features)), random_state=random_state + 1)

    artifacts = {
        "background": background.reset_index(drop=True),
        "sample": sample.reset_index(drop=True),
        "feature_columns": list(features.columns),
    }

    joblib.dump(artifacts, output_dir / "shap_background.joblib")
    (output_dir / "shap_sample.csv").write_text(sample.to_csv(index=False))

    print(f"Saved SHAP artifacts to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SHAP background and sample artifacts.")
    parser.add_argument("--data", default="data/processed/engineered_features.csv")
    parser.add_argument("--out", default="models")
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    prepare_shap_inputs(
        Path(args.data),
        Path(args.out),
        background_size=args.background_size,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
