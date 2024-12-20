"""
This script takes as input a GeoDataFrame of edges and write a CSV or Parquet file corresponding to
the METROPOLIS2 format.
"""

import os
import time

import polars as pl

import metropy.utils.io as metro_io


def read_edges(
    edge_filename: str,
    edge_main_filename: None | str,
    edge_penalties_filename: None | str,
    edge_capacities_filename: None | str,
):
    print("Reading edges")
    edges = metro_io.scan_dataframe(edge_filename)
    if edge_main_filename is not None:
        edges_main = metro_io.scan_dataframe(edge_main_filename).filter("main")
        edges = edges.join(edges_main, on=pl.col("edge_id"), how="semi")
    if edge_penalties_filename is not None:
        edges_penalties = metro_io.scan_dataframe(edge_penalties_filename).rename(
            {"additive_penalty": "constant_travel_time"}
        )
        columns = edges_penalties.collect_schema().names()
        assert (
            "constant_travel_time" in columns
        ), "No column `additive_penalty` in the edges penalties file"
        if "speed" in columns:
            edges = edges.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("constant_travel_time").fill_null(pl.lit(0.0)),
                pl.col("speed").fill_null(pl.col("speed_limit")),
            )
        else:
            print("Warning: no `speed` column in the edges penalties, using additive penalty only")
            edges = edges.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("constant_travel_time").fill_null(pl.lit(0.0)),
                pl.col("speed_limit").alias("speed"),
            )
    else:
        edges = edges.with_columns(constant_travel_time=pl.lit(0.0), speed=pl.col("speed_limit"))
    if edge_capacities_filename is not None:
        edge_capacities = metro_io.scan_dataframe(edge_capacities_filename).select(
            "edge_id", "capacity"
        )
        edges = edges.join(edge_capacities, on="edge_id", how="left")
    return edges


def generate_edges(edges: pl.LazyFrame, config: dict, remove_parallel=True):
    print("Creating METROPOLIS edges...")
    # Convert edges' speed from km/h to m/s.
    edges = edges.with_columns(pl.col("speed") / 3.6)
    edges = edges.with_columns(pl.lit(config.get("overtaking", True)).alias("overtaking"))
    columns = [
        "edge_id",
        "source",
        "target",
        "speed",
        "length",
        "lanes",
        "overtaking",
        "constant_travel_time",
    ]
    sort_columns = ["lanes", "speed", "length"]
    sort_descending = [True, True, False]
    if config.get("use_bottleneck", False):
        edges = edges.with_columns(bottleneck_flow=pl.col("capacity") / 3600)
        columns.append("bottleneck_flow")
        sort_columns.insert(0, "bottleneck_flow")
        sort_descending.insert(0, True)
    edges_df = edges.select(columns).collect()
    if remove_parallel:
        # Remove parallel edges.
        n0 = len(edges_df)
        edges_df = edges_df.sort(sort_columns, descending=sort_descending).unique(
            subset=["source", "target"], keep="first"
        )
        n1 = len(edges_df)
        if n0 > n1:
            print("Warning: Discarded {:,} parallel edges".format(n0 - n1))
    edges_df = edges_df.sort("source")
    return edges_df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "run_directory",
    ]
    check_keys(config, mandatory_keys)
    t0 = time.time()

    if not os.path.isdir(os.path.join(config["run_directory"], "input")):
        os.makedirs(os.path.join(config["run_directory"], "input"))

    edges = read_edges(
        config["clean_edges_file"],
        config.get("routing", dict).get("road_split", dict).get("main_edges_filename"),
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
        config.get("capacities_filename"),
    )

    edges = generate_edges(edges, config["run"])

    print("Writing edges")
    edges.write_parquet(os.path.join(config["run_directory"], "input", "edges.parquet"))

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
