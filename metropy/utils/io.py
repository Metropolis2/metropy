import os

import polars as pl
import geopandas as gpd
from pyogrio.errors import DataSourceError


def scan_dataframe(filename: str, **kwargs):
    """Scan a DataFrame from a Parquet or CSV file."""
    if not os.path.isfile(filename):
        raise Exception(f"File not found: `{filename}`")
    if filename.endswith(".parquet"):
        lf = pl.read_parquet(filename, use_pyarrow=True, **kwargs).lazy()
    elif filename.endswith(".csv"):
        lf = pl.scan_csv(filename, **kwargs)
    else:
        raise Exception(f"Unsupported format for input file: `{filename}`")
    return lf


def read_dataframe(filename: str, columns=None, **kwargs):
    """Reads a DataFrame from a Parquet or CSV file."""
    lf = scan_dataframe(filename, **kwargs)
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


def read_geodataframe(filename: str, columns=None):
    """Reads a GeoDataFrame from a Parquet file or any other format supported by GeoPandas."""
    if not os.path.isfile(filename):
        raise Exception(f"File not found: `{filename}`")
    if filename.endswith(".parquet"):
        gdf = gpd.read_parquet(filename, columns=columns)
    else:
        try:
            gdf = gpd.GeoDataFrame(gpd.read_file(filename, columns=columns,
                                                                   engine="pyogrio"))
        except DataSourceError:
            raise Exception(f"Unsupported format for input file: `{filename}`")
    return gdf


def save_dataframe(df: pl.DataFrame, filename: str):
    """Saves a DataFrame to a Parquet or CSV file."""
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if filename.endswith("parquet"):
        df.write_parquet(filename, use_pyarrow=True)
    elif filename.endswith("csv"):
        df.write_csv(filename)
    else:
        raise Exception(f"Unsupported format for output file: `{filename}`")


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
