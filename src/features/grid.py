import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml
from pathlib import Path
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_grid(config: dict) -> gpd.GeoDataFrame:
    bbox = config["region"]["bbox"]
    res = config["data"]["resolution"]

    lats = np.arange(bbox["south"], bbox["north"], res)
    lons = np.arange(bbox["west"], bbox["east"], res)

    cells = []
    for lat in lats:
        for lon in lons:
            cells.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "geometry": Point(lon, lat)
            })

    gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    logger.success(f"Created grid with {len(gdf)} cells")
    return gdf


if __name__ == "__main__":
    config = load_config(
        config_path=str(Path(__file__).resolve().parents[2] / "configs/config.yaml")
    )
    grid = create_grid(config)
    print(grid.head())
    print(f"Total grid cells: {len(grid)}")