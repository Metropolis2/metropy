import os
import time

import pandas as pd
import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io


def read_trips(directory: str):
    print("Reading trips...")
    df = metro_io.read_dataframe(
        os.path.join(directory, "trips.parquet"),
        columns=["trip_id", "origin_lng", "origin_lat", "destination_lng", "destination_lat"],
    )
    origins = gpd.GeoDataFrame(
        data={"trip_id": df["trip_id"]},
    )
    origins.set_geometry(
        gpd.points_from_xy(df["origin_lng"], df["origin_lat"], crs="epsg:4326"), inplace=True
    )
    destinations = gpd.GeoDataFrame(
        data={"trip_id": df["trip_id"]},
    )
    destinations.set_geometry(
        gpd.points_from_xy(df["destination_lng"], df["destination_lat"], crs="epsg:4326"),
        inplace=True,
    )
    return origins, destinations


def read_iris(filename: str):
    print("Reading IRIS zones...")
    gdf = metro_io.read_geodataframe(filename, columns=["CODE_IRIS", "geometry"])
    gdf.rename(columns={"CODE_IRIS": "iris"}, inplace=True)
    return gdf


def get_zones(origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, iris: gpd.GeoDataFrame):
    assert iris.crs is not None
    print("Finding origin IRIS zones...")
    origins.to_crs(iris.crs, inplace=True)
    origins = origins.sjoin(iris, predicate="within", how="left")
    origins.drop_duplicates(subset=["trip_id"], inplace=True)
    print("Finding destination IRIS zones...")
    destinations.to_crs(iris.crs, inplace=True)
    destinations = destinations.sjoin(iris, predicate="within", how="left")
    destinations.drop_duplicates(subset=["trip_id"], inplace=True)
    print("Merging results...")
    zones_pd = pd.merge(
        origins[["trip_id", "iris"]],
        destinations[["trip_id", "iris"]],
        on="trip_id",
        suffixes=("_origin", "_destination"),
    )
    zones = pl.from_pandas(zones_pd).with_columns(
        pl.col("iris_origin").str.slice(0, 5).alias("insee_origin"),
        pl.col("iris_destination").str.slice(0, 5).alias("insee_destination"),
        pl.col("iris_origin").str.slice(0, 2).alias("departement_origin"),
        pl.col("iris_destination").str.slice(0, 2).alias("departement_destination"),
    )
    return zones


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "crs",
        "population_directory",
        "france.iris_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()
    origins, destinations = read_trips(config["population_directory"])
    iris = read_iris(config["france"]["iris_filename"])
    zones = get_zones(origins, destinations, iris)
    metro_io.save_dataframe(
        zones, os.path.join(config["population_directory"], "trip_zones.parquet")
    )
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
