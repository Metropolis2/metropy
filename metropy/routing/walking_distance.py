import os
import shutil
import time
import json
import subprocess

import numpy as np
import polars as pl
import geopandas as gpd
from shapely.geometry import Point

import metropy.utils.mpl as mpl


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
    # Create a DataFrame will all the unique (lng, lat) pairs.
    nodes = pl.concat(
        (
            df.select(pl.col("origin_lng").alias("lng"), pl.col("origin_lat").alias("lat")),
            df.select(
                pl.col("destination_lng").alias("lng"), pl.col("destination_lat").alias("lat")
            ),
        ),
        how="vertical",
    ).unique()
    print(f"Number of unique origin / destination points: {len(nodes)}")
    return df, nodes


def read_edges(filename: str, crs: str, config: dict):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(
        filename, columns=["id", "source", "target", "length", "road_type", "geometry"]
    )
    gdf.rename(columns={"id": "edge_id"}, inplace=True)
    gdf.to_crs(crs, inplace=True)
    if "forbidden_road_types" in config:
        assert isinstance(config["forbidden_road_types"], list)
        gdf["allow_od"] = ~gdf["road_type"].isin(config["forbidden_road_types"])
    else:
        gdf["allow_od"] = True
    return gdf


def find_origin_destination_node(nodes: pl.DataFrame, edges: gpd.GeoDataFrame):
    print("Creating origin / destination points...")
    # Create a GeoDataFrame of all the origin / destination nodes.
    gdf = gpd.GeoDataFrame(
        data={"lng": nodes["lng"], "lat": nodes["lat"]},
        geometry=gpd.points_from_xy(nodes["lng"], nodes["lat"], crs="EPSG:4326"),
    )
    gdf.to_crs(edges.crs, inplace=True)
    print("Matching to the nearest edge...")
    # Match to the nearest edge.
    gdf = gdf.sjoin_nearest(
        edges.loc[edges["allow_od"], ["edge_id", "geometry", "source", "target"]],
        distance_col="edge_dist",
        how="left",
    )
    # Duplicate indices occur when there are two edges at the same distance.
    gdf.drop_duplicates(subset=["lng", "lat"], inplace=True)
    print("Creating source / target points...")
    # Create source / target point of the edges.
    matched_edges = edges.loc[edges["edge_id"].isin(gdf["edge_id"]), ["edge_id", "geometry"]]
    matched_edges["source_point"] = matched_edges["geometry"].apply(lambda g: Point(g.coords[0]))
    matched_edges["target_point"] = matched_edges["geometry"].apply(lambda g: Point(g.coords[-1]))
    gdf = gdf.merge(matched_edges[["edge_id", "source_point", "target_point"]], on="edge_id")
    print("Computing distances to source / target nodes...")
    # Compute distance to the source / target node of nearest edge.
    gdf["source_dist"] = gdf["geometry"].distance(gdf["source_point"])
    gdf["target_dist"] = gdf["geometry"].distance(gdf["target_point"])
    print("Finding nearest node...")
    # Set the nearest node.
    df = pl.from_pandas(
        gdf.loc[
            :, ["lng", "lat", "edge_id", "edge_dist", "source", "target", "source_dist", "target_dist"]
        ]
    )
    mask = df["source_dist"] > df["target_dist"]
    df = df.with_columns(
        pl.when(mask).then(pl.col("target")).otherwise(pl.col("source")).alias("node"),
        pl.when(mask)
        .then(pl.col("target_dist"))
        .otherwise(pl.col("source_dist"))
        .alias("node_dist"),
    )
    print(
        "Distance between origin / destination point and nearest edge: {}".format(
            df["edge_dist"].describe()
        )
    )
    print(
        "Distance between origin / destination point and nearest node: {}".format(
            df["node_dist"].describe()
        )
    )
    df = df.select(
        "lng",
        "lat",
        "node",
        "node_dist",
        "edge_dist",
    )
    return df


def add_od_to_trips(trips: pl.DataFrame, df: pl.DataFrame):
    print("Assigning origin / destination node to all trips...")
    lztrips = (
        trips.lazy()
        .join(df.lazy(), left_on=["origin_lng", "origin_lat"], right_on=["lng", "lat"])
        .rename(
            {
                "node": "origin_node",
                "node_dist": "origin_node_dist",
                "edge_dist": "origin_edge_dist",
            }
        )
    )
    trips = (
        lztrips.join(
            df.lazy(), left_on=["destination_lng", "destination_lat"], right_on=["lng", "lat"]
        )
        .rename(
            {
                "node": "destination_node",
                "node_dist": "destination_node_dist",
                "edge_dist": "destination_edge_dist",
            }
        )
        .select(
            "trip_id",
            "origin_node",
            "origin_node_dist",
            "origin_edge_dist",
            "destination_node",
            "destination_node_dist",
            "destination_edge_dist",
        )
        .collect()
    )
    return trips


def prepare_routing(trips: pl.DataFrame, edges: gpd.GeoDataFrame, tmp_directory: str):
    print("Saving queries...")
    queries = trips.unique(subset=["origin_node", "destination_node"]).select(
        pl.int_range(0, pl.len(), dtype=pl.UInt64).alias("query_id"),
        pl.col("origin_node").alias("origin"),
        pl.col("destination_node").alias("destination"),
        pl.lit(0.0).alias("departure_time"),
    )
    print("Number of unique origin nodes: {:,}".format(queries["origin"].n_unique()))
    print("Number of unique destination nodes: {:,}".format(queries["destination"].n_unique()))
    print("Number of unique OD pairs: {:,}".format(len(queries)))
    queries.write_parquet(os.path.join(tmp_directory, "queries.parquet"))
    print("Saving graph...")
    edges = pl.from_pandas(edges.loc[:, ["edge_id", "source", "target", "length"]])
    # Parallel edges are removed, keeping only the shortest edge.
    edges.sort("length").unique(subset=["source", "target"], keep="first").sort("edge_id").select(
        "edge_id", "source", "target", pl.col("length").alias("travel_time")
    ).write_parquet(os.path.join(tmp_directory, "edges.parquet"))
    print("Saving parameters...")
    parameters = {
        "algorithm": "Best",
        "output_route": False,
        "input_files": {
            "queries": "queries.parquet",
            "edges": "edges.parquet",
        },
        "output_directory": "output",
        "saving_format": "Parquet",
    }
    with open(os.path.join(tmp_directory, "parameters.json"), "w") as f:
        json.dump(parameters, f)
    return queries


def run_routing(routing_exec: str, tmp_directory: str):
    # Create the routing command.
    parameters_filename = os.path.join(tmp_directory, "parameters.json")
    command = f"{routing_exec} {parameters_filename}"
    print("Running routing...")
    subprocess.run(command, shell=True)


def load_results(tmp_directory: str):
    print("Reading routing results...")
    return pl.read_parquet(os.path.join(tmp_directory, "output", "ea_results.parquet"))


def merge(trips: pl.DataFrame, queries: pl.DataFrame, results: pl.DataFrame):
    print("Merging trips data...")
    results = results.join(queries, on="query_id").select(
        "origin", "destination", pl.col("arrival_time").alias("distance_network")
    )
    trips = trips.join(
        results,
        left_on=["origin_node", "destination_node"],
        right_on=["origin", "destination"],
        how="left",
    )
    print("Walking distance (network):\n{}".format(trips["distance_network"].describe()))
    trips = trips.with_columns(
        (
            pl.col("distance_network")
            + pl.col("origin_node_dist")
            + pl.col("destination_node_dist")
        ).alias("distance")
    )
    print("Walking distance (total):\n{}".format(trips["distance"].describe()))
    return trips


def plot_variables(trips: pl.DataFrame, graph_dir: str):
    print("Generating graphs of the variables")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    # Distance between origin and start node.
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(0.0, np.log(trips.select("origin_node_dist").min().item()))
    bins = (
        np.logspace(m, np.log1p(trips.select("origin_node_dist").max().item()), 50, base=np.e)
        - 1.0
    )
    ax.hist(trips["origin_node_dist"], bins=bins, color=mpl.CMP(0))
    ax.set_xlabel("Distance origin / start node (meters)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "origin_node_dist_distribution.pdf"))
    # Distance between destination and end node.
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(0.0, np.log(trips.select("destination_node_dist").min().item()))
    bins = (
        np.logspace(
            m, np.log1p(trips.select("destination_node_dist").max().item()), 50, base=np.e
        )
        - 1.0
    )
    ax.hist(trips["destination_node_dist"], bins=bins, color=mpl.CMP(0))
    ax.set_xlabel("Distance destination / end node (meters)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "destination_node_dist_distribution.pdf"))
    # Walking distance network.
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(0.0, np.log(trips.select("distance_network").min().item()))
    bins = (
        np.logspace(m, np.log1p(trips.select("distance_network").max().item()), 50, base=np.e)
        - 1.0
    )
    ax.hist(
        trips.filter(pl.col("distance_network") > 0)["distance_network"],
        bins=bins,
        color=mpl.CMP(1),
    )
    ax.set_xlabel("Walking distance on network (meters)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "walking_distance_network_distribution.pdf"))
    # Walking distance.
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(0.0, np.log(trips.select("distance").min().item()))
    bins = np.logspace(m, np.log1p(trips.select("distance").max().item()), 50, base=np.e) - 1.0
    ax.hist(trips.filter(pl.col("distance") > 0)["distance"], bins=bins, color=mpl.CMP(1))
    ax.set_xlabel("Walking distance (meters)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "walking_distance_distribution.pdf"))


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

    trips, nodes = read_trips(config["population_directory"])

    edges = read_edges(
        config["clean_walk_edges_file"], config["crs"], config["routing"]["walking_distance"]
    )

    df = find_origin_destination_node(nodes, edges)

    trips = add_od_to_trips(trips, df)

    queries = prepare_routing(trips, edges, config["tmp_directory"])

    run_routing(config["routing_exec"], config["tmp_directory"])

    results = load_results(config["tmp_directory"])

    trips = merge(trips, queries, results)

    metro_io.save_dataframe(
        trips,
        config["routing"]["walking_distance"]["output_filename"],
    )

    if config["routing"]["walking_distance"].get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "routing.walking_distance")
        plot_variables(trips, graph_dir)

    # Clean the temporary directory.
    try:
        shutil.rmtree(config["tmp_directory"])
    except OSError as e:
        print(e)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
