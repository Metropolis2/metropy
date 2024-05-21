import os
import time
import json
import subprocess

import polars as pl
import geopandas as gpd
from shapely.geometry import Point

def read_trips(input_directory: str):
    print("Reading trips' coordinates...")
    columns = [
        "trip_id",
        "origin_lng",
        "origin_lat",
        "destination_lng",
        "destination_lat",
    ]
    df = pl.read_parquet(
        os.path.join(input_directory, "trips.parquet"),
        columns=columns,
    )
    df = df.filter(
        (pl.col("origin_lng") != pl.col("destination_lng"))
        | (pl.col("origin_lat") != pl.col("destination_lat"))
    )
    return df


def read_edges(filename: str, crs: str, config: dict):
    print("Reading edges")
    gdf = metro_io.read_geodataframe(filename, columns=["id", "source", "target", "length"])
    gdf.to_crs(crs, inplace=True)
    if "forbidden_road_types" in config:
        assert isinstance(config["forbidden_road_types"], list)
        gdf["allow_od"] = ~gdf["road_type"].isin(config["forbidden_road_types"])
    else:
        gdf["allow_od"] = True
    return gdf


def find_origin_destination_node(trips, edges):
    print("Finding closest edge to each origin")
    # Create source point of each edge.
    edges["source_point"] = edges["geometry"].apply(lambda g: Point(g.coords[0]))
    # Find the closest edge to each origin, with the distance to the edge and to the source node.
    origins = gpd.GeoDataFrame(
        {"id": trips["trip_id"]},
        geometry=gpd.points_from_xy(trips["origin_lng"], trips["origin_lat"], crs="EPSG:4326"),
    )
    origins.sjoin_nearest(
        edges.loc[edges["allow_od"], ["source", "geometry", "source_point"]],
        distance_col="origin_edge_dist",
        how="left",
    )
    # Duplicate indices occur when there are two edges at the same distance.
    origins = origins.drop_duplicates(subset=["trip_id"])
    origins["origin_source_dist"] = origins.geometry.distance(origins["source_point"])
    trips = trips.join(
        pl.from_pandas(origins[["trip_id", "source", "origin_edge_dist", "origin_source_dist"]]),
        on="trip_id",
    ).rename({"source": "origin_node"})
    # Compute the distance to the source node when traveling through the edge, using Pythagore
    # Theorem.
    trips = trips.with_columns(
        ((pl.col("origin_source_dist") ** 2) - (pl.col("origin_edge_dist") ** 2))
        .sqrt()
        .alias("origin_source_dist_on_edge")
    )
    print("Finding closest edge to each destination")
    # Do the same for the destinations.
    destinations = gpd.GeoDataFrame(
        {"id": trips["trip_id"]},
        geometry=gpd.points_from_xy(
            trips["destination_lng"], trips["destination_lat"], crs="EPSG:4326"
        ),
    )
    destinations.sjoin_nearest(
        edges.loc[edges["allow_od"], ["source", "geometry", "source_point"]],
        distance_col="destination_edge_dist",
        how="left",
    )
    destinations = destinations.drop_duplicates(subset=["trip_id"])
    destinations["destination_source_dist"] = destinations.geometry.distance(
        destinations["source_point"]
    )
    trips = trips.join(
        pl.from_pandas(
            destinations[["trip_id", "source", "destination_edge_dist", "destination_source_dist"]]
        ),
        on="trip_id",
    ).rename({"source": "destination_node"})
    trips = trips.with_columns(
        ((pl.col("destination_source_dist") ** 2) - (pl.col("destination_edge_dist") ** 2))
        .sqrt()
        .alias("destination_source_dist_on_edge")
    )
    print("Number of unique origin nodes: {:,}".format(trips["origin_node"].n_unique()))
    print("Number of unique destination nodes: {:,}".format(trips["destination_node"].n_unique()))
    n = trips.n_unique(subset=["origin_node", "destination_node"])
    print("Number of unique OD pairs: {:,}".format(n))
    print(
        "Distance between origin point and start edge: {}".format(
            trips["origin_edge_dist"].describe()
        )
    )
    print(
        "Distance between destination point and end edge: {}".format(
            trips["destination_edge_dist"].describe()
        )
    )
    trips = trips.select(
        "trip_id",
        "origin_node",
        "destination_node",
        "origin_edge_dist",
        "origin_source_dist",
        "origin_source_dist_on_edge",
        "destination_edge_dist",
        "destination_source_dist",
        "destination_source_dist_on_edge",
    )
    return trips


def prepare_routing(trips: pl.DataFrame, edges, tmp_directory: str):
    print("Saving queries")
    queries = trips.unique(subset=["origin_node", "destination_node"]).select(
        pl.int_range(0, pl.len(), dtype=pl.UInt64).alias("query_id"),
        pl.col("node_origin_walk").alias("origin"),
        pl.col("node_destination_walk").alias("destination"),
        pl.lit(0.0).alias("departure_time"),
    )
    queries.write_parquet(os.path.join(tmp_directory, "queries.parquet"))
    print("Saving graph")
    # Parallel edges are removed, keeping only the shortest edge.
    edges.sort("length").unique(subset=["source", "target"], keep="first").sort("id").select(
        pl.col("id").alias("edge_id"), "source", "target", pl.col("length").alias("travel_time")
    ).write_parquet(os.path.join(tmp_directory, "edges.parquet"))
    print("Saving parameters")
    parameters = {
        "algorithm": "Best",
        "output_route": False,
        "input_files": {
            "queries": "queries.parquet",
            "edges": "edges.parquet",
        },
        "output_directory": os.path.join(tmp_directory, "output"),
        "saving_format": "Parquet",
    }
    with open(os.path.join(tmp_directory, "parameters.json"), "w") as f:
        json.dump(parameters, f)
    return queries


def run_routing(routing_exec: str, tmp_directory: str):
    # Create the routing command.
    parameters_filename = os.path.join(tmp_directory, "parameters.json")
    command = f"{routing_exec} {parameters_filename}"
    print("Running routing")
    subprocess.run(command, shell=True)


def load_results(tmp_directory: str):
    print("Reading routing results")
    return pl.read_parquet(os.path.join(tmp_directory, "output", "ea_results.parquet"))


def merge(trips: pl.DataFrame, queries: pl.DataFrame, results: pl.DataFrame):
    print("Merging trips data")
    results = results.join(queries, on="query_id").select(
        "origin", "destination", pl.col("arrival_time").alias("distance")
    )
    trips = trips.join(
        results,
        left_on=["origin_node", "destination_node"],
        right_on=["origin", "destination"],
        how="left",
    )
    print("Walking distance:\n{}".format(trips["distance"].describe()))
    return trips


if __name__ == "__main__":
    from metropy.config import read_config, check_keys
    import metropy.utils.io as metro_io

    config = read_config()
    mandatory_keys = [
        "clean_walk_edges_file",
        "population_directory",
        "crs",
        "routing_exec",
        "tmp_directory",
        "routing.walking_distance",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()

    trips = read_trips(config["population_directory"])

    edges = read_edges(
        config["clean_walk_edges_file"], config["crs"], config["routing"]["walking_distance"]
    )

    df = find_origin_destination_node(trips, edges)

    queries = prepare_routing(df, edges, config["tmp_directory"])

    run_routing(config["routing_exec"], config["tmp_directory"])

    results = load_results(config["tmp_directry"])

    df = merge(df, queries, results)

    metro_io.save_dataframe(
        df,
        config["routing"]["walking_distance"]["output_filename"],
    )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
