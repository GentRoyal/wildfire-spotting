import cdsapi
import yaml
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_era5(year: int, month: int, config: dict):
    client = cdsapi.Client()
    bbox = config["region"]["bbox"]
    raw_path = Path(config["paths"]["raw"]) / "era5"
    raw_path.mkdir(parents=True, exist_ok=True)

    zip_path = raw_path / f"era5_{year}_{month:02d}.zip"
    instant_path = raw_path / f"era5_{year}_{month:02d}_instant.nc"
    accum_path = raw_path / f"era5_{year}_{month:02d}_accum.nc"

    if instant_path.exists() and accum_path.exists():
        logger.info(f"Already exists, skipping: {year}-{month:02d}")
        return

    logger.info(f"Downloading ERA5 for {year}-{month:02d}...")

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation",
                "surface_pressure",
                "volumetric_soil_water_layer_1",
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [
                bbox["north"],
                bbox["west"],
                bbox["south"],
                bbox["east"],
            ],
            "data_format": "netcdf",
        },
        str(zip_path),
    )

    logger.info("Extracting downloaded file...")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        for name in names:
            if "instant" in name:
                extracted = raw_path / name
                z.extract(name, raw_path)
                extracted.rename(instant_path)
            elif "accum" in name:
                extracted = raw_path / name
                z.extract(name, raw_path)
                extracted.rename(accum_path)

    zip_path.unlink()
    logger.success(f"Saved: {instant_path.name} and {accum_path.name}")


if __name__ == "__main__":
    config = load_config()

    # Download 2020-2022
    for year in range(2020, 2026):
        for month in range(1, 13):
            download_era5(year=year, month=month, config=config)