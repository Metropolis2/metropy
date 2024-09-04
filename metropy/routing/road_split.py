import os
import shutil
import time
import json
import subprocess
import gc

import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import networkx as nx
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
    ).with_columns(is_truck=pl.lit(False))
    truck_filename = os.path.join(input_directory, "truck_trips.parquet")
    if os.path.isfile(truck_filename):
        truck_df = pl.read_parquet(truck_filename, columns=columns).with_columns(
            is_truck=pl.lit(True)
        )
        df = pl.concat((df, truck_df), how="vertical")
        assert df["trip_id"].n_unique() == len(
            df
        ), "Column `trip_id` must be unique over agent and truck trips"
    n0 = len(df)
    df = df.filter(
        (pl.col("origin_lng") != pl.col("destination_lng"))
        | (pl.col("origin_lat") != pl.col("destination_lat"))
    )
    n1 = len(df)
    if n0 > n1:
        print("Warning: discarding {:,} trips with origin equal to destination".format(n0 - n1))
    # Create a DataFrame will all the unique (lng, lat) pairs.
    nodes = pl.concat(
        (
            df.select(
                pl.col("origin_lng").alias("lng"), pl.col("origin_lat").alias("lat"), "is_truck"
            ),
            df.select(
                pl.col("destination_lng").alias("lng"),
                pl.col("destination_lat").alias("lat"),
                "is_truck",
            ),
        ),
        how="vertical",
    ).unique()
    print(f"Number of unique origin / destination points: {len(nodes):,}")
    return df, nodes


def read_edges(
    filename: str,
    crs: str,
    config: dict,
    edge_penalties_filename: str | None,
    truck_speed_limit: float | None,
):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(
        filename,
        columns=["edge_id", "source", "target", "length", "speed_limit", "road_type", "geometry"],
    )
    gdf.to_crs(crs, inplace=True)
    if "forbidden_road_types" in config:
        gdf["allow_od"] = ~gdf["road_type"].isin(config["forbidden_road_types"])
    else:
        gdf["allow_od"] = True
    if "truck_forbidden_road_types" in config:
        gdf["allow_truck_od"] = ~gdf["road_type"].isin(config["truck_forbidden_road_types"])
    else:
        gdf["allow_truck_od"] = True
    gdf["main"] = gdf["road_type"].isin(config["main_road_types"])
    if edge_penalties_filename is not None:
        penalties = metro_io.read_dataframe(edge_penalties_filename)
        if "speed" in penalties.columns:
            print("Using multiplicative penalties")
            gdf = gpd.GeoDataFrame(
                gdf.merge(
                    penalties.select("edge_id", "speed").to_pandas(),
                    on="edge_id",
                    how="left",
                )
            )
            gdf["travel_time_car"] = gdf["length"] / (gdf["speed"] / 3.6)
        else:
            gdf["travel_time_car"] = gdf["length"] / (gdf["speed_limit"] / 3.6)
        if "additive_penalty" in penalties.columns:
            print("Using additive penalties")
            gdf = gpd.GeoDataFrame(
                gdf.merge(
                    penalties.select("edge_id", "additive_penalty").to_pandas(),
                    on="edge_id",
                    how="left",
                )
            )
            gdf["travel_time_car"] += gdf["additive_penalty"].fillna(0.0)
    else:
        gdf["travel_time_car"] = gdf["length"] / (gdf["speed_limit"] / 3.6)
    if not truck_speed_limit:
        truck_speed_limit = np.inf
    gdf["travel_time_truck"] = np.minimum(gdf["travel_time_car"], truck_speed_limit)
    return gdf


def process_edges(edges: pl.DataFrame):
    print("Finding the largest strongly connected component of the main graph...")
    G = nx.DiGraph()
    G.add_edges_from(
        map(
            lambda v: (v[0], v[1]),
            edges.filter(pl.col("main")).select("source", "target").to_numpy(),
        )
    )
    # Find the nodes of the largest strongly connected component.
    nodes = max(nx.strongly_connected_components(G), key=len)
    if len(nodes) < G.number_of_nodes():
        print(
            """Warning: discarding {} nodes from the main graph as they are disconnected from the
            largest component of the main graph""".format(
                G.number_of_nodes() - len(nodes)
            )
        )
        n0 = edges["main"].sum()
        l0 = edges.filter(pl.col("main"))["length"].sum()
        edges = edges.with_columns(
            (pl.col("main") & pl.col("source").is_in(nodes) & pl.col("target").is_in(nodes)).alias(
                "main"
            )
        )
        n1 = edges["main"].sum()
        l1 = edges.filter(pl.col("main"))["length"].sum()
        print("Number of edges discarded: {} ({:.2%})".format(n0 - n1, (n0 - n1) / n0))
        print("Edge length discarded (m): {:,.0f} ({:.2%})".format(l0 - l1, (l0 - l1) / l0))
    return edges


def find_origin_destination_node(nodes: pl.DataFrame, edges: gpd.GeoDataFrame):
    print("Creating origin / destination points...")
    # Create a GeoDataFrame of all the origin / destination nodes.
    gdf = gpd.GeoDataFrame(
        data={"lng": nodes["lng"], "lat": nodes["lat"], "is_truck": nodes["is_truck"]},
        geometry=gpd.points_from_xy(nodes["lng"], nodes["lat"], crs="EPSG:4326"),
    )
    gdf.to_crs(edges.crs, inplace=True)
    print("Matching to the nearest edge...")
    # Match to the nearest edge.
    gdf = pd.concat(
        (
            gdf.loc[~gdf["is_truck"]].sjoin_nearest(
                edges.loc[edges["allow_od"], ["edge_id", "geometry", "source", "target"]],
                distance_col="edge_dist",
                how="left",
            ),
            gdf.loc[gdf["is_truck"]].sjoin_nearest(
                edges.loc[edges["allow_truck_od"], ["edge_id", "geometry", "source", "target"]],
                distance_col="edge_dist",
                how="left",
            ),
        )
    )
    # Duplicate indices occur when there are two edges at the same distance.
    gdf.drop_duplicates(subset=["lng", "lat", "is_truck"], inplace=True)
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
            :,
            [
                "lng",
                "lat",
                "is_truck",
                "edge_id",
                "edge_dist",
                "source",
                "target",
                "source_dist",
                "target_dist",
            ],
        ],
        schema_overrides={"edge_id": pl.UInt64, "source": pl.UInt64, "target": pl.UInt64},
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
        "is_truck",
    )
    return df


def add_od_to_trips(trips: pl.DataFrame, df: pl.DataFrame):
    print("Assigning origin / destination node to all trips...")
    n = len(trips)
    lztrips = (
        trips.lazy()
        .join(
            df.lazy(),
            left_on=["origin_lng", "origin_lat", "is_truck"],
            right_on=["lng", "lat", "is_truck"],
        )
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
            df.lazy(),
            left_on=["destination_lng", "destination_lat", "is_truck"],
            right_on=["lng", "lat", "is_truck"],
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
            "is_truck",
            "origin_node",
            "origin_node_dist",
            "origin_edge_dist",
            "destination_node",
            "destination_node_dist",
            "destination_edge_dist",
        )
        .collect()
    )
    assert len(trips) == n
    return trips


def run_routing(
    trips: pl.DataFrame, edges: gpd.GeoDataFrame, routing_exec: str, tmp_directory: str
):
    if not trips["is_truck"].all():
        parameters_filename = prepare_routing(
            trips.filter(~pl.col("is_truck")), edges, tmp_directory, "car"
        )
        print("Running routing for car...")
        subprocess.run(" ".join([routing_exec, parameters_filename]), shell=True)
    if trips["is_truck"].any():
        parameters_filename = prepare_routing(
            trips.filter(pl.col("is_truck")), edges, tmp_directory, "truck"
        )
        print("Running routing for truck...")
        subprocess.run(" ".join([routing_exec, parameters_filename]), shell=True)


def prepare_routing(trips: pl.DataFrame, edges: gpd.GeoDataFrame, tmp_directory: str, suffix: str):
    print(f"Saving queries for {suffix}...")
    queries = (
        trips.unique(subset=["origin_node", "destination_node"])
        .filter(pl.col("origin_node") != pl.col("destination_node"))
        .select(
            pl.int_range(0, pl.len(), dtype=pl.UInt64).alias("query_id"),
            pl.col("origin_node").alias("origin"),
            pl.col("destination_node").alias("destination"),
            pl.lit(0.0).alias("departure_time"),
        )
    )
    print(f"Number of unique origin-destination pairs: {len(queries):,}")
    queries.write_parquet(os.path.join(tmp_directory, f"queries_{suffix}.parquet"))
    print("Saving graph...")
    edges_df = pl.from_pandas(
        edges.loc[:, ["edge_id", "source", "target", f"travel_time_{suffix}", "main"]]
    )
    # Parallel edges are removed, keeping in priority edges of the main graph and with smallest
    # travel time.
    edges_df.sort(["main", f"travel_time_{suffix}"], descending=[True, False]).unique(
        subset=["source", "target"], keep="first"
    ).rename({f"travel_time_{suffix}": "travel_time"}).sort("edge_id").select(
        "edge_id", "source", "target", "travel_time"
    ).write_parquet(
        os.path.join(tmp_directory, f"edges_{suffix}.parquet")
    )
    print("Saving parameters...")
    parameters = {
        "algorithm": "TCH",
        "output_route": True,
        "input_files": {
            "queries": f"queries_{suffix}.parquet",
            "edges": f"edges_{suffix}.parquet",
        },
        "output_directory": f"output_{suffix}",
        "saving_format": "Parquet",
    }
    parameters_filename = os.path.join(tmp_directory, f"parameters_{suffix}.json")
    with open(parameters_filename, "w") as f:
        json.dump(parameters, f)
    return parameters_filename


def load_results(tmp_directory: str):
    results = pl.DataFrame()
    car_filename = os.path.join(tmp_directory, "output_car", "ea_results.parquet")
    if os.path.isfile(car_filename):
        print("Reading routing results for car...")
        results = pl.read_parquet(car_filename).with_columns(is_truck=pl.lit(False))
    truck_filename = os.path.join(tmp_directory, "output_truck", "ea_results.parquet")
    if os.path.isfile(truck_filename):
        print("Reading routing results for truck...")
        results = pl.concat(
            (results, pl.read_parquet(truck_filename).with_columns(is_truck=pl.lit(True))),
            how="vertical",
        )
    assert not results.is_empty()
    return results


def edges_to_polars(edges_gdf: gpd.GeoDataFrame):
    edges = pl.from_pandas(
        edges_gdf.loc[
            :,
            [
                "edge_id",
                "source",
                "target",
                "travel_time_car",
                "travel_time_truck",
                "length",
                "main",
            ],
        ],
        schema_overrides={"edge_id": pl.UInt64, "source": pl.UInt64, "target": pl.UInt64},
    )
    return edges


def find_main_edges(results: pl.DataFrame, edges: pl.DataFrame):
    main_edges = set(edges.filter(pl.col("main"))["edge_id"])
    return find_main_edges_inner(results, main_edges)


def find_main_edges_inner(results: pl.DataFrame, main_edges: set, i=0):
    print(f"=== Iteration {i} ===")
    # Compute the set of edges taken during the main part (i.e., in between the first and last main
    # edge, inclusive).
    edges_in_main_part = (
        results.lazy()
        .with_columns(
            pl.col("route").list.eval(pl.element().is_in(main_edges)).alias("route_main_mask")
        )
        .filter(pl.col("route_main_mask").list.any())
        .with_columns(
            pl.col("route_main_mask").list.arg_max().alias("first_idx_main"),
            (
                pl.col("route").list.len() - pl.col("route_main_mask").list.reverse().list.arg_max()
            ).alias("last_idx_main"),
        )
        .with_columns(
            pl.col("route")
            .list.slice(
                offset=pl.col("first_idx_main"),
                length=pl.col("last_idx_main") - pl.col("first_idx_main"),
            )
            .alias("route_main_part"),
        )
        .select("route_main_part")
        .collect()
    )
    all_edges = set(
        edges_in_main_part.select(pl.col("route_main_part").list.explode().unique()).to_series()
    )
    del edges_in_main_part
    gc.collect()
    middle_secondaries = all_edges.difference(main_edges)
    if middle_secondaries:
        print(
            "{:,} edges from secondary graph are added to the main graph".format(
                len(middle_secondaries)
            )
        )
        main_edges = main_edges.union(middle_secondaries)
        return find_main_edges_inner(results, main_edges, i + 1)
    print("Total number of edges in the main graph: {:,}".format(len(main_edges)))
    return main_edges


def get_main_edges(edges: pl.DataFrame, main_edges: set):
    # Flag main edges (used in Metropolis).
    edges = edges.with_columns(pl.col("edge_id").is_in(main_edges).alias("main"))
    print("Finding the largest strongly connected component of the main graph...")
    G = nx.DiGraph()
    G.add_edges_from(
        map(
            lambda v: (v[0], v[1]),
            edges.filter(pl.col("main")).select("source", "target").to_numpy(),
        )
    )
    assert nx.is_strongly_connected(G)
    return edges.select("edge_id", "main")


def find_connections(results: pl.DataFrame, edges: pl.DataFrame, main_edges: set[int]):
    print("Finding access / egress part...")
    lazy_results = (
        results.lazy()
        .rename({"arrival_time": "free_flow_travel_time"})
        .with_columns(
            pl.col("route").list.eval(pl.element().is_in(main_edges)).alias("route_main_mask")
        )
        .with_columns((~pl.col("route_main_mask").list.any()).alias("secondary_only"))
        .with_columns(
            pl.col("route")
            .list.first()
            .replace_strict(edges["edge_id"], edges["source"], return_dtype=pl.UInt64)
            .alias("origin_node"),
            pl.col("route")
            .list.last()
            .replace_strict(edges["edge_id"], edges["target"], return_dtype=pl.UInt64)
            .alias("destination_node"),
        )
    )
    results_main = (
        lazy_results.filter(~pl.col("secondary_only"))
        .with_columns(
            pl.col("route_main_mask").list.arg_max().alias("first_idx_main"),
            (
                pl.col("route").list.len() - pl.col("route_main_mask").list.reverse().list.arg_max()
            ).alias("last_idx_main"),
        )
        .with_columns(
            pl.col("route")
            .list.slice(
                offset=0,
                length=pl.col("first_idx_main"),
            )
            .alias("access_part"),
            pl.col("route").list.slice(offset=pl.col("last_idx_main")).alias("egress_part"),
        )
        .with_columns(
            pl.col("route")
            .list.get(pl.col("first_idx_main"))
            .replace_strict(edges["edge_id"], edges["source"], return_dtype=pl.UInt64)
            .alias("access_node"),
            pl.col("route")
            .list.get(pl.col("last_idx_main") - 1)
            .replace_strict(edges["edge_id"], edges["target"], return_dtype=pl.UInt64)
            .alias("egress_node"),
            pl.when("is_truck")
            .then(
                pl.col("access_part")
                .list.eval(
                    pl.element().replace_strict(
                        edges["edge_id"], edges["travel_time_truck"], return_dtype=pl.UInt64
                    )
                )
                .list.sum()
            )
            .otherwise(
                pl.col("access_part")
                .list.eval(
                    pl.element().replace_strict(
                        edges["edge_id"], edges["travel_time_car"], return_dtype=pl.UInt64
                    )
                )
                .list.sum()
            )
            .alias("access_time"),
            pl.when("is_truck")
            .then(
                pl.col("egress_part")
                .list.eval(
                    pl.element().replace_strict(
                        edges["edge_id"], edges["travel_time_truck"], return_dtype=pl.UInt64
                    )
                )
                .list.sum()
            )
            .otherwise(
                pl.col("egress_part")
                .list.eval(
                    pl.element().replace_strict(
                        edges["edge_id"], edges["travel_time_car"], return_dtype=pl.UInt64
                    )
                )
                .list.sum()
            )
            .alias("egress_time"),
        )
        .select(
            "access_part",
            "egress_part",
            "access_node",
            "egress_node",
            "origin_node",
            "access_time",
            "egress_time",
            "destination_node",
            "route",
            "free_flow_travel_time",
            "secondary_only",
            "is_truck",
        )
    )
    results_secondary = lazy_results.filter(pl.col("secondary_only")).select(
        "route",
        "origin_node",
        "destination_node",
        "free_flow_travel_time",
        "secondary_only",
        "is_truck",
    )
    # TODO: save edges with count and main columns.
    results = pl.concat((results_main.collect(), results_secondary.collect()), how="diagonal")
    return results


def merge(trips: pl.DataFrame, results: pl.DataFrame):
    print("Merging trips data...")
    trips = (
        trips.lazy()
        .join(
            results.lazy(),
            on=["origin_node", "destination_node", "is_truck"],
            how="left",
            coalesce=False,
        )
        .with_columns(
            pl.col("free_flow_travel_time").fill_null(0.0), pl.col("secondary_only").fill_null(True)
        )
        .select(
            "trip_id",
            "origin_node",
            "origin_node_dist",
            "origin_edge_dist",
            "destination_node",
            "destination_node_dist",
            "destination_edge_dist",
            "access_node",
            "egress_node",
            "access_part",
            "egress_part",
            "access_time",
            "egress_time",
            "free_flow_travel_time",
            "route",
            "secondary_only",
            "is_truck",
        )
        .collect()
    )
    print(
        "Share of trips not taking the main network: {:.2%}".format(trips["secondary_only"].mean())
    )
    print("Free-flow travel time:\n{}".format(trips["free_flow_travel_time"].describe()))
    print("Access time:\n{}".format(trips["access_time"].describe()))
    print("Egress time:\n{}".format(trips["egress_time"].describe()))
    print("Number of unique access nodes: {:,}".format(trips["access_node"].n_unique()))
    print("Number of unique egress nodes: {:,}".format(trips["egress_node"].n_unique()))
    n = trips.n_unique(subset=["access_node", "egress_node"])
    print(f"Number of unique access / egress pairs: {n:,}")
    secondary_tt = (
        trips.lazy()
        .select(
            pl.when(pl.col("secondary_only"))
            .then(pl.col("free_flow_travel_time"))
            .otherwise(pl.col("access_time") + pl.col("egress_time"))
            .sum()
        )
        .collect()
        .item()
        / 3600
    )
    tot_tt = trips["free_flow_travel_time"].sum() / 3600
    print(f"Total free-flow travel time (hours): {tot_tt:,.0f}")
    print(
        "Total free-flow travel time on secondary edges (hours): {:,.0f} ({:.2%})".format(
            secondary_tt, secondary_tt / tot_tt
        )
    )
    return trips


#  def edge_counts(df: pl.DataFrame):
#  access_counts = (
#  df.lazy()
#  .filter(pl.col("access_part").list.len().gt(0))
#  .select(pl.col("access_part").explode().value_counts())
#  .unnest("access_part")
#  .rename({"access_part": "edge_id"})
#  )
#  egress_counts = (
#  df.lazy()
#  .filter(pl.col("egress_part").list.len().gt(0))
#  .select(pl.col("egress_part").explode().value_counts())
#  .unnest("egress_part")
#  .rename({"egress_part": "edge_id"})
#  )
#  secondary_counts = (
#  df.lazy()
#  .filter(pl.col("route").list.len().gt(0))
#  .select(pl.col("route").explode().value_counts())
#  .unnest("route")
#  .rename({"route": "edge_id"})
#  )
#  counts = (
#  access_counts.join(egress_counts, on="edge_id", how="full", coalesce=True)
#  .select(
#  "edge_id",
#  (pl.col("count").fill_null(0) + pl.col("count_right").fill_null(0)).alias("count"),
#  )
#  .join(secondary_counts, on="edge_id", how="full", coalesce=True)
#  .select(
#  "edge_id",
#  (pl.col("count").fill_null(0) + pl.col("count_right").fill_null(0)).alias("count"),
#  )
#  .collect()
#  )


def plot_variables(trips: pl.DataFrame, graph_dir: str):
    print("Generating graphs of the variables")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    # Distance between origin and start node.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = (
        np.logspace(0.0, np.log1p(trips.select("origin_node_dist").max().item()), 50, base=np.e)
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
    bins = (
        np.logspace(
            0.0, np.log1p(trips.select("destination_node_dist").max().item()), 50, base=np.e
        )
        - 1.0
    )
    ax.hist(trips["destination_node_dist"], bins=bins, color=mpl.CMP(0))
    ax.set_xlabel("Distance destination / end node (meters)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "destination_node_dist_distribution.pdf"))
    # Free-flow travel time.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = (
        np.logspace(
            0.0, np.log1p(trips.select("free_flow_travel_time").max().item()), 50, base=np.e
        )
        - 1.0
    )
    ax.hist(
        trips.filter(pl.col("free_flow_travel_time") > 0)["free_flow_travel_time"] / 60,
        bins=bins / 60,
        color=mpl.CMP(1),
    )
    ax.set_xlabel("Free-flow travel time (minutes)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "free_flow_travel_time_distribution.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys
    import metropy.utils.io as metro_io

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "population_directory",
        "crs",
        "routing_exec",
        "tmp_directory",
        "routing.road_split.main_road_types",
        "routing.road_split.trips_filename",
        "routing.road_split.main_edges_filename",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()

    trips, nodes = read_trips(config["population_directory"])

    edges = read_edges(
        config["clean_edges_file"],
        config["crs"],
        config["routing"]["road_split"],
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
        config.get("run", dict).get("truck_speed_limit"),
    )

    df = find_origin_destination_node(nodes, edges)

    trips = add_od_to_trips(trips, df)

    edges_df = edges_to_polars(edges)

    edges_df = process_edges(edges_df)

    run_routing(trips, edges, config["routing_exec"], config["tmp_directory"])

    results = load_results(config["tmp_directory"])

    main_edges = find_main_edges(results, edges_df)

    results = find_connections(results, edges_df, main_edges)

    main_edges_df = get_main_edges(edges_df, main_edges)

    df = merge(trips, results)

    metro_io.save_dataframe(
        df,
        config["routing"]["road_split"]["trips_filename"],
    )

    metro_io.save_dataframe(
        main_edges_df,
        config["routing"]["road_split"]["main_edges_filename"],
    )

    if config["routing"]["road_split"].get("output_graphs", False):
        if "graph_directory" not in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "routing.road_split")
        plot_variables(df, graph_dir)

    # Clean the temporary directory.
    try:
        shutil.rmtree(config["tmp_directory"])
    except OSError as e:
        print(e)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
