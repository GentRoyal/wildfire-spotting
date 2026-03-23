import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_era5_month(year: int, month: int, raw_path: Path) -> xr.Dataset:
    instant_path = raw_path / f"era5_{year}_{month:02d}_instant.nc"
    accum_path = raw_path / f"era5_{year}_{month:02d}_accum.nc"

    if not instant_path.exists() or not accum_path.exists():
        logger.warning(f"Missing ERA5 files for {year}-{month:02d}")
        return None

    ds_instant = xr.open_dataset(instant_path, engine="netcdf4")
    ds_accum = xr.open_dataset(accum_path, engine="netcdf4")
    return xr.merge([ds_instant, ds_accum])


def extract_weather_for_grid(
    ds: xr.Dataset,
    grid: pd.DataFrame,
    timestamp: pd.Timestamp
) -> pd.DataFrame:
    # Select nearest timestamp in dataset
    ds_t = ds.sel(valid_time=timestamp, method="nearest")

    records = []
    for _, cell in grid.iterrows():
        lat, lon = cell["lat"], cell["lon"]

        # Select nearest grid point
        point = ds_t.sel(latitude=lat, longitude=lon, method="nearest")

        t2m = float(point["t2m"].values) - 273.15         # Celsius
        d2m = float(point["d2m"].values) - 273.15         # Dewpoint Celsius
        u10 = float(point["u10"].values)                   # Wind U component
        v10 = float(point["v10"].values)                   # Wind V component
        sp = float(point["sp"].values)                     # Surface pressure
        swvl1 = float(point["swvl1"].values)               # Soil moisture
        tp = float(point["tp"].values) * 1000              # Precip mm

        # Derived variables
        wind_speed = np.sqrt(u10**2 + v10**2)
        rh = 100 * (np.exp((17.625 * d2m) / (243.04 + d2m)) /
                    np.exp((17.625 * t2m) / (243.04 + t2m)))  # Relative humidity %
        vpd = (1 - rh / 100) * 6.1078 * np.exp(17.27 * t2m / (t2m + 237.3))  # Vapor pressure deficit

        records.append({
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "t2m": round(t2m, 3),
            "d2m": round(d2m, 3),
            "wind_speed": round(wind_speed, 3),
            "wind_u": round(u10, 3),
            "wind_v": round(v10, 3),
            "surface_pressure": round(sp, 3),
            "soil_moisture": round(swvl1, 4),
            "precipitation_mm": round(tp, 4),
            "relative_humidity": round(rh, 2),
            "vpd": round(vpd, 3),
        })

    return pd.DataFrame(records)


def build_weather_dataset(
    grid: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    raw_path: Path,
    config: dict
) -> pd.DataFrame:
    all_frames = []
    current_ds = None
    current_year_month = None

    for ts in timestamps:
        year, month = ts.year, ts.month
        ym = (year, month)

        if ym != current_year_month:
            logger.info(f"Loading ERA5 for {year}-{month:02d}...")
            current_ds = load_era5_month(year, month, raw_path)
            current_year_month = ym

        if current_ds is None:
            logger.warning(f"Skipping {ts} — no ERA5 data")
            continue

        df = extract_weather_for_grid(current_ds, grid, ts)
        all_frames.append(df)
        logger.info(f"Extracted features for {ts}")

    combined = pd.concat(all_frames, ignore_index=True)
    logger.success(f"Weather dataset built: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


if __name__ == "__main__":
    from grid import create_grid
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    config = load_config(
        config_path=str(Path(__file__).resolve().parents[2] / "configs/config.yaml")
    )
    grid = create_grid(config)
    raw_path = Path(__file__).resolve().parents[2] / "data/raw/era5"

    # Test on one week first
    timestamps = pd.date_range("2022-07-01", "2022-07-07", freq="6h")

    weather_df = build_weather_dataset(grid, timestamps, raw_path, config)

    output_path = Path(__file__).resolve().parents[2] / "data/processed/weather_test.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(output_path, index=False)

    print(weather_df.head())
    print(weather_df.describe())