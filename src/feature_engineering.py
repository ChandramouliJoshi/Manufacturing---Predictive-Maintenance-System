"""Feature engineering for machine sensor time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


SENSOR_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


@dataclass(frozen=True)
class FeatureConfig:
    input_path: Path = Path("data/raw/ai4i2020.csv")
    output_path: Path = Path("data/processed/engineered_features.csv")
    metadata_path: Path = Path("models/feature_engineering.joblib")
    rolling_windows: tuple[int, ...] = (1, 6, 12)
    lag_steps: tuple[int, ...] = (1, 2)


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    return pd.read_csv(path)


def create_time_series_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    engineered = df.copy()
    available_sensors = [col for col in SENSOR_COLUMNS if col in engineered.columns]

    for column in available_sensors:
        for window in config.rolling_windows:
            rolling = engineered[column].rolling(window=window, min_periods=1)
            engineered[f"{column}_roll_mean_{window}h"] = rolling.mean()
            engineered[f"{column}_roll_std_{window}h"] = rolling.std().fillna(0)
            engineered[f"{column}_ema_{window}h"] = (
                engineered[column].ewm(span=window, adjust=False).mean()
            )

        for lag in config.lag_steps:
            engineered[f"{column}_lag_t{lag}"] = engineered[column].shift(lag)

        engineered[f"{column}_delta_1h"] = engineered[column].diff()

    engineered = engineered.bfill().ffill()
    return engineered


def save_feature_artifacts(
    engineered: pd.DataFrame,
    config: FeatureConfig,
    source_columns: list[str],
) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    engineered.to_csv(config.output_path, index=False)
    metadata = {
        "source_columns": source_columns,
        "engineered_columns": list(engineered.columns),
        "sensor_columns": [col for col in SENSOR_COLUMNS if col in source_columns],
        "rolling_windows_hours": list(config.rolling_windows),
        "lag_steps": list(config.lag_steps),
    }
    joblib.dump(metadata, config.metadata_path)


def build_features(config: FeatureConfig) -> pd.DataFrame:
    raw = load_raw_data(config.input_path)
    engineered = create_time_series_features(raw, config)
    save_feature_artifacts(engineered, config, list(raw.columns))
    return engineered
