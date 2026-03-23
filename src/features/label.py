import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml
from grid import create_grid


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_firms(raw_path: Path) -> pd.DataFrame:
    all_files = list(raw_path.glob("firms_*.csv"))
    frames = []
    for f in all_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["acq_date"] = pd.to_datetime(combined["acq_date"])
    logger.success(f"Loaded {len(combined):,} fire detections")
    return combined


def assign_fire_labels(
    grid: gpd.GeoDataFrame,
    fires: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: dict,
    window_hours: int = 72
) -> pd.DataFrame:
    res = config["data"]["resolution"]
    records = []

    logger.info(f"Assigning labels for {len(timestamps)} timestamps across {len(grid)} grid cells...")

    for ts in timestamps:
        window_end = ts + pd.Timedelta(hours=window_hours)
        fires_in_window = fires[
            (fires["acq_date"] >= ts) &
            (fires["acq_date"] < window_end)
        ]

        for _, cell in grid.iterrows():
            lat, lon = cell["lat"], cell["lon"]

            # Check if any fire falls within this grid cell
            fire_in_cell = fires_in_window[
                (fires_in_window["latitude"].between(lat, lat + res)) &
                (fires_in_window["longitude"].between(lon, lon + res))
            ]

            records.append({
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "fire_within_72h": int(len(fire_in_cell) > 0)
            })

    df = pd.DataFrame(records)
    pos = df["fire_within_72h"].sum()
    total = len(df)
    logger.success(
        f"Labels assigned: {pos:,} positive ({pos/total*100:.2f}%), "
        f"{total-pos:,} negative out of {total:,} total"
    )
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    config = load_config(
        config_path=str(Path(__file__).resolve().parents[2] / "configs/config.yaml")
    )
    grid = create_grid(config)

    raw_path = Path(__file__).resolve().parents[2] / "data/raw/firms"

    # Check files exist before loading
    csv_files = list(raw_path.glob("firms_*.csv"))
    logger.info(f"Found {len(csv_files)} FIRMS files in {raw_path}")

    if not csv_files:
        logger.error(f"No FIRMS files found at {raw_path}. Run firms_ingestion.py first.")
        sys.exit(1)

    fires = load_firms(raw_path)

    # Test on a small window first
    timestamps = pd.date_range("2022-07-01", "2022-07-07", freq="6h")
    labels = assign_fire_labels(grid, fires, timestamps, config)

    output_path = Path(__file__).resolve().parents[2] / "data/processed/labels_test.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    print(labels["fire_within_72h"].value_counts())