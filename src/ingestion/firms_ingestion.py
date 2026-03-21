import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os
import yaml

load_dotenv()

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_firms(year: int, month: int, config: dict):
    api_key = os.getenv("FIRMS_API_KEY")
    bbox = config["region"]["bbox"]
    raw_path = Path(config["paths"]["raw"]) / "firms"
    raw_path.mkdir(parents=True, exist_ok=True)

    output_path = raw_path / f"firms_{year}_{month:02d}.csv"

    if output_path.exists():
        logger.info(f"Already exists, skipping: {output_path.name}")
        return

    area = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = [d for d in days if d.month == month]

    all_frames = []

    for day in days:
        date_str = day.strftime("%Y-%m-%d")
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{api_key}/VIIRS_SNPP_SP/{area}/1/{date_str}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            logger.warning(f"Failed for {date_str}: {response.status_code}")
            continue

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        if len(df) > 0:
            all_frames.append(df)
            logger.info(f"{date_str}: {len(df)} detections")
        else:
            logger.info(f"{date_str}: 0 detections")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(output_path, index=False)
        logger.success(f"Saved {len(combined)} total detections to {output_path.name}")
    else:
        logger.warning("No fire detections found for this period")
if __name__ == "__main__":
    config = load_config()
    download_firms(year=2022, month=7, config=config)