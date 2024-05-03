import os

import geopandas as gpd
from fiona.errors import DriverError


def read_geodataframe(filename: str, columns=None):
    """Reads a GeoDataFrame from a Parquet file or any other format supported by GeoPandas."""
    if not os.path.isfile(filename):
        raise Exception(f"File not found: `{filename}`")
    if filename.endswith(".parquet"):
        gdf = gpd.read_parquet(filename, columns=columns)
    else:
        try:
            gdf = gpd.read_file(filename, columns=columns)
        except DriverError:
            raise Exception(f"Unsupported format for input file: `{filename}`")
    return gdf


def save_geodataframe(gdf: gpd.GeoDataFrame, filename: str):
    """Saves a GeoDataFrame to a Parquet, GeoJSON, FlatGeobuf or Shapefile file."""
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if filename.endswith("parquet"):
        gdf.to_parquet(filename)
    elif filename.endswith("geojson"):
        gdf.to_file(filename, driver="GeoJSON")
    elif filename.endswith("fgb"):
        gdf.to_file(filename, driver="FlatGeobuf")
    elif filename.endswith("shp"):
        gdf.to_file(filename, driver="Shapefile")
    else:
        raise Exception(f"Unsupported format for output file: `{filename}`")
