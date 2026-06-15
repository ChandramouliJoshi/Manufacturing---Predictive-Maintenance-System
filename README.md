# Manufacturing Predictive Maintenance System

This project follows the feature-engineering and modeling plan from the
requirements image:

- advanced rolling-window statistics over 1, 6, and 12 hour windows
- exponential moving averages
- standard deviation features
- lag features for t-1 and t-2
- joblib serialization for feature metadata and trained models
- imbalance-aware model training focused on PR-AUC, precision, and recall

## Run Feature Engineering

```bash
python main.py features
```

This creates:

- `data/processed/engineered_features.csv`
- `models/feature_engineering.joblib`

## Train Models

Train the production XGBoost model:

```bash
python main.py train --model xgboost
```

Run feature engineering and modeling together:

```bash
python main.py all --model xgboost
```

Baseline models are also available:

```bash
python main.py --model random_forest
python main.py --model logistic_regression
```

## Explainability (SHAP)

Generate SHAP artifacts from the engineered feature dataset:

```bash
python main.py explain --shap-out models --background-size 100 --sample-size 20
```

Compute and save SHAP values plus plot images after training:

```bash
python main.py explain --compute-shap --shap-out models --shap-plot-out outputs/plots
```

The explainability artifacts are stored in `models/shap_background.joblib`, `models/shap_sample.csv`, and `models/shap_values.joblib`, while plot images are written to the configured `outputs/plots` directory.

LightGBM is available for a second production-grade option:

```bash
python main.py all --model lightgbm
```

The default target is `machine_failure`. Failure-mode columns such as `twf`,
`hdf`, `pwf`, `osf`, `rnf`, plus `failure_24h`, are excluded during training to
avoid leakage.

The metrics report is saved to:

- `outputs/reports/imbalance_model_metrics.json`

Accuracy is intentionally not used as the main metric because failures are rare.
The training report prioritizes PR-AUC, precision, recall, and F1.
