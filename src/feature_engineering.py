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
    """Create exactly the 79 features expected by the trained model."""
    engineered = df.copy()
    
    # Add 'type' column first (required by model - column 1)
    if 'type' not in engineered.columns:
        engineered.insert(0, 'type', 0)
    
    # Ensure all sensor columns present (columns 2-6)
    available_sensors = [col for col in SENSOR_COLUMNS if col in engineered.columns]
    
    # Sensor short-name map
    sensor_map = {
        "air_temperature_k": "temp",
        "process_temperature_k": "process_temp",
        "rotational_speed_rpm": "rpm",
        "torque_nm": "torque",
        "tool_wear_min": "tool_wear",
    }
    
    # Pre-compute rolling windows (span=1, 6, 12 hours)
    rolling_data = {}
    for col in available_sensors:
        rolling_data[col] = {}
        for window in config.rolling_windows:
            rolling = engineered[col].rolling(window=window, min_periods=1)
            rolling_data[col][window] = {
                'mean': rolling.mean(),
                'std': rolling.std().fillna(0),
                'ema': engineered[col].ewm(span=window, adjust=False).mean()
            }
    
    # Short-name features (columns 7-19, specific selection matching training data)
    # These were hand-selected during original training
    engineered["temp_roll_mean_6"] = rolling_data["air_temperature_k"][6]["mean"]
    engineered["temp_roll_mean_12"] = rolling_data["air_temperature_k"][12]["mean"]
    engineered["torque_std_6"] = rolling_data["torque_nm"][6]["std"]
    engineered["rpm_std_6"] = rolling_data["rotational_speed_rpm"][6]["std"]
    engineered["temp_ema_6"] = rolling_data["air_temperature_k"][6]["ema"]
    engineered["torque_ema_6"] = rolling_data["torque_nm"][6]["ema"]
    engineered["temp_lag1"] = engineered["air_temperature_k"].shift(1)
    engineered["temp_lag2"] = engineered["air_temperature_k"].shift(2)
    engineered["torque_lag1"] = engineered["torque_nm"].shift(1)
    engineered["temp_delta"] = engineered["air_temperature_k"].diff()
    engineered["torque_delta"] = engineered["torque_nm"].diff()
    engineered["rpm_change"] = engineered["rotational_speed_rpm"].diff()
    engineered["torque_change"] = engineered["torque_nm"].diff()
    
    # Full-name features (columns 20-79, systematic across all 5 sensors)
    for col in available_sensors:
        for window in config.rolling_windows:
            engineered[f"{col}_roll_mean_{window}h"] = rolling_data[col][window]["mean"]
            engineered[f"{col}_roll_std_{window}h"] = rolling_data[col][window]["std"]
            engineered[f"{col}_ema_{window}h"] = rolling_data[col][window]["ema"]
        
        # Lags
        for lag in config.lag_steps:
            engineered[f"{col}_lag_t{lag}"] = engineered[col].shift(lag)
        
        # Delta
        engineered[f"{col}_delta_1h"] = engineered[col].diff()
    
    # Fill NaN values from rolling/diff operations
    engineered = engineered.bfill().ffill()
    
    # Reorder columns to exact training order: type + sensors + short-names + full-names
    # First 6 columns: type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min
    base_cols = ["type"] + available_sensors
    
    # Short-name features (columns 7-19)
    short_feature_cols = [
        "temp_roll_mean_6", "temp_roll_mean_12", "torque_std_6", "rpm_std_6",
        "temp_ema_6", "torque_ema_6", "temp_lag1", "temp_lag2", "torque_lag1",
        "temp_delta", "torque_delta", "rpm_change", "torque_change"
    ]
    
    # Full-name features (columns 20-79)
    full_feature_cols = []
    for col in available_sensors:
        for window in [1, 6, 12]:  # Order matters: 1h, 6h, 12h
            full_feature_cols.append(f"{col}_roll_mean_{window}h")
            full_feature_cols.append(f"{col}_roll_std_{window}h")
            full_feature_cols.append(f"{col}_ema_{window}h")
        for lag in [1, 2]:
            full_feature_cols.append(f"{col}_lag_t{lag}")
        full_feature_cols.append(f"{col}_delta_1h")
    
    # Combine in exact order
    final_cols = base_cols + short_feature_cols + full_feature_cols
    engineered = engineered[final_cols]
    
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
