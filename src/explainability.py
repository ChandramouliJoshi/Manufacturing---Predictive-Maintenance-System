"""Prepare SHAP inputs and save background/sample artifacts for explainability."""

from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import pandas as pd


LEAKAGE_COLUMNS = {
    "machine_failure",
    "failure_24h",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
}


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def prepare_shap_inputs(
    data_path: Path,
    output_dir: Path,
    background_size: int = 100,
    sample_size: int = 20,
    random_state: int = 42,
):
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    features = df[feature_cols].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    background = features.sample(
        n=min(background_size, len(features)), random_state=random_state
    ).reset_index(drop=True)
    sample = features.sample(
        n=min(sample_size, len(features)), random_state=random_state + 1
    ).reset_index(drop=True)

    artifacts = {
        "background": background,
        "sample": sample,
        "feature_columns": list(feature_cols),
    }

    joblib.dump(artifacts, output_dir / "shap_background.joblib")
    (output_dir / "shap_sample.csv").write_text(sample.to_csv(index=False), encoding="utf-8")

    print(f"Saved SHAP artifacts to {output_dir}")
    return artifacts


def compute_shap_values(
    model,
    background: pd.DataFrame,
    sample: pd.DataFrame,
    feature_cols: list[str],
):
    import shap

    def predict_probabilities(data):
        return model.predict_proba(pd.DataFrame(data, columns=feature_cols))[:, 1]

    explainer = shap.Explainer(predict_probabilities, background, feature_names=feature_cols)
    return explainer(sample)


def save_shap_values(shap_values, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "shap_explanation": shap_values,
        "values": shap_values.values,
        "base_values": shap_values.base_values,
        "data": shap_values.data,
        "feature_names": list(shap_values.feature_names),
    }
    joblib.dump(artifact, output_dir / "shap_values.joblib")
    print(f"Saved SHAP values to {output_dir / 'shap_values.joblib'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SHAP background and sample artifacts for explainability.")
    parser.add_argument("--data", default="data/processed/engineered_features.csv")
    parser.add_argument("--model", default="models/production_failure_model.joblib")
    parser.add_argument("--out", default="models")
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute SHAP values for the saved sample set.",
    )
    args = parser.parse_args()

    artifacts = prepare_shap_inputs(
        Path(args.data),
        Path(args.out),
        background_size=args.background_size,
        sample_size=args.sample_size,
    )

    if args.compute_shap:
        model = load_model(Path(args.model))
        shap_values = compute_shap_values(
            model,
            artifacts["background"],
            artifacts["sample"],
            artifacts["feature_columns"],
        )
        save_shap_values(shap_values, Path(args.out))


if __name__ == "__main__":
    main()
