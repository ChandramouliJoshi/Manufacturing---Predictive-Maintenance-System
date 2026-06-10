# Project Content Till Now

## Project Name

Manufacturing Predictive Maintenance System

## Project Goal

The project predicts machine failure using manufacturing sensor data. The main focus so far is on feature engineering, imbalance-aware model training, and evaluation using metrics suitable for rare failure events.

## Dataset

The project uses the dataset:

```text
data/raw/ai4i2020.csv
```

The dataset contains machine sensor and failure-related columns such as:

- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`
- `machine_failure`
- failure mode columns like `twf`, `hdf`, `pwf`, `osf`, and `rnf`

## Feature Engineering Completed

Feature engineering has been implemented in:

```text
src/feature_engineering.py
```

The implemented features include:

- Rolling mean features for 1, 6, and 12 hour windows
- Rolling standard deviation features for 1, 6, and 12 hour windows
- Exponential moving average features for 1, 6, and 12 hour windows
- Lag features for `t-1` and `t-2`
- One-step delta/change features
- Joblib serialization for feature engineering metadata

The generated feature output is saved to:

```text
data/processed/engineered_features.csv
```

Feature metadata is saved to:

```text
models/feature_engineering.joblib
```

## Modeling Completed

Model training has been implemented in:

```text
src/modeling.py
```

The modeling pipeline includes:

- Logistic Regression baseline
- Random Forest baseline
- XGBoost production model
- LightGBM production model
- Class imbalance handling using class weights or `scale_pos_weight`
- Threshold selection focused on high precision
- Leakage prevention by removing failure-mode columns from training

The default target column is:

```text
machine_failure
```

## Evaluation Metrics

Since machine failure is rare, accuracy is not used as the main metric. The project focuses on:

- PR-AUC
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report

The latest XGBoost run produced:

```text
PR-AUC: 0.7717
Precision: 0.8077
Recall: 0.6176
F1 score: 0.7000
```

The metrics report is saved to:

```text
outputs/reports/imbalance_model_metrics.json
```

## Main Pipeline

The main command-line pipeline is implemented in:

```text
main.py
```

Available commands:

```bash
python main.py features
python main.py train --model xgboost
python main.py all --model xgboost
python main.py all --model lightgbm
```

## Files Updated or Added

The following project files have been created or updated:

- `main.py`
- `src/__init__.py`
- `src/feature_engineering.py`
- `src/modeling.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `project contnet.md`

## Current Status

The project currently has a working feature engineering and imbalance-aware modeling pipeline. The XGBoost model has been trained and verified successfully, and the output metrics show strong precision-focused performance for rare machine failure prediction.
