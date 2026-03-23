import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.grid import create_grid
from features.weather_features import build_weather_dataset
from features.label import load_firms, assign_fire_labels


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing anomaly features...")

    df = df.sort_values(["lat", "lon", "timestamp"]).reset_index(drop=True)

    for var in ["t2m", "relative_humidity", "wind_speed", "vpd", "soil_moisture"]:
        # 7-day rolling mean and std per grid cell
        df[f"{var}_roll7d_mean"] = (
            df.groupby(["lat", "lon"])[var]
            .transform(lambda x: x.rolling(window=28, min_periods=1).mean())
        )
        df[f"{var}_roll7d_std"] = (
            df.groupby(["lat", "lon"])[var]
            .transform(lambda x: x.rolling(window=28, min_periods=1).std().fillna(1))
        )
        # Z-score: how anomalous is current value vs recent baseline
        df[f"{var}_zscore"] = (
            (df[var] - df[f"{var}_roll7d_mean"]) / df[f"{var}_roll7d_std"]
        )

    # Consecutive dry timesteps (precip < 0.1mm)
    df["is_dry"] = (df["precipitation_mm"] < 0.1).astype(int)
    df["consecutive_dry_steps"] = (
        df.groupby(["lat", "lon"])["is_dry"]
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        * df["is_dry"]
    )

    # Rapid humidity drop (change from previous timestep)
    df["rh_drop"] = df.groupby(["lat", "lon"])["relative_humidity"].diff().clip(upper=0).abs()

    # Fosberg Fire Weather Index (FFWI) — simplified
    df["ffwi"] = (
        df["wind_speed"] *
        ((1 - (df["relative_humidity"] / 100)) ** 2) *
        (df["t2m"] / 30).clip(lower=0.1)
    ).round(4)

    # Clamp relative humidity
    df["relative_humidity"] = df["relative_humidity"].clip(0, 100)

    logger.success(f"Anomaly features added. Dataset now has {len(df.columns)} columns")
    return df


def build_full_dataset(
    start_date: str,
    end_date: str,
    config: dict,
    raw_era5_path: Path,
    raw_firms_path: Path,
    output_path: Path
):
    grid = create_grid(config)
    timestamps = pd.date_range(start_date, end_date, freq="6h")

    # Weather features
    logger.info("Building weather feature dataset...")
    weather_df = build_weather_dataset(grid, timestamps, raw_era5_path, config)

    # Fire labels
    logger.info("Loading fire detections...")
    fires = load_firms(raw_firms_path)
    labels_df = assign_fire_labels(grid, fires, timestamps, config)

    # Join
    logger.info("Joining weather features and fire labels...")
    dataset = weather_df.merge(
        labels_df[["timestamp", "lat", "lon", "fire_within_72h"]],
        on=["timestamp", "lat", "lon"],
        how="left"
    )
    dataset["fire_within_72h"] = dataset["fire_within_72h"].fillna(0).astype(int)

    # Add anomaly features
    dataset = add_anomaly_features(dataset)

    # Drop intermediate rolling columns
    cols_to_drop = [c for c in dataset.columns if "roll7d" in c or c == "is_dry"]
    dataset = dataset.drop(columns=cols_to_drop)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    logger.success(f"Dataset saved: {len(dataset):,} rows x {len(dataset.columns)} columns")
    logger.info(f"Class balance — Positive: {dataset['fire_within_72h'].sum():,} "
                f"({dataset['fire_within_72h'].mean()*100:.2f}%)")

    return dataset


if __name__ == "__main__":
    config = load_config(
        config_path=str(Path(__file__).resolve().parents[2] / "configs/config.yaml")
    )
    base = Path(__file__).resolve().parents[2]

    dataset = build_full_dataset(
        start_date="2020-01-01",
        end_date="2025-12-31",
        config=config,
        raw_era5_path=base / "data/raw/era5",
        raw_firms_path=base / "data/raw/firms",
        output_path=base / "data/processed/dataset_2020_2025.parquet"
    )

    print(dataset.head())
    print(f"\nColumns: {dataset.columns.tolist()}")
    print(f"\nShape: {dataset.shape}")