"""
This script takes as input a GeoDataFrame of edges and write a CSV or Parquet file corresponding to
the METROPOLIS2 format.
"""

import sys
import os
import time
import tomllib

import polars as pl

import metropy.utils.io as metro_io


def read_edges(input_file):
    print("Reading edges")
    edges = pl.from_pandas(metro_io.read_geodataframe(input_file).drop(columns="geometry"))
    if "main" in edges.columns:
        edges = edges.filter("main")
    else:
        print("Warning: no 'main' column in edges, selecting all edges")
    edges = edges.sort("edge_id")
    return edges


def generate_road_network(edges):
    print("Creating Metropolis road network")
    edges = edges.with_columns(pl.col("speed_limit") / 3.6)
    edges = edges.with_columns(pl.lit(True).alias("overtaking"))
    columns = ["edge_id", "source", "target", "speed_limit", "length", "lanes", "overtaking"]
    if "capacity" in edges.columns:
        edges = edges.with_columns((pl.col("capacity") / 3600.0).alias("bottleneck_flow"))
        columns.append("bottleneck_flow")
    # TODO: Add penalties
    edges = edges.select(columns)
    return edges


if __name__ == "__main__":
    t0 = time.time()

    if len(sys.argv) == 1:
        # Read the config from the default path.
        config_path = "config.toml"
    else:
        if len(sys.argv) == 3 and sys.argv[1] == "--config":
            config_path = sys.argv[2]
        else:
            raise SystemExit(f"Usage: {sys.argv[0]} [--config <path_to_config.toml>]")

    if not os.path.exists(config_path):
        raise Exception(f"Cannot find config file `{config_path}`")

    with open(config_path, "rb") as f:
        try:
            config = tomllib.load(f)
        except Exception as e:
            raise Exception(f"Cannot parse config:\n{e}")

    input_file = config.get("clean_edges_file")
    if input_file is None:
        raise Exception("Missing key `clean_edges_file` in config")
    if not os.path.exists(input_file):
        raise Exception(f"Edges input file not found:\n`{input_file}`")

    if not "metropolis" in config:
        raise Exception("Missing table `metropolis` in config")
    run_dir = config["metropolis"].get("input_directory")
    if run_dir is None:
        raise Exception("Missing key `metropolis.input_directory` in config")
    input_format = config["metropolis"].get("format", "Parquet")

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    edges = read_edges(input_file)

    edges = generate_road_network(edges)

    print("Writing edges")
    if input_format.lower() == "parquet":
        edges.write_parquet(os.path.join(run_dir, "edges.parquet"))
    elif input_format.lower() == "csv":
        edges.write_csv(os.path.join(run_dir, "edges.csv"))
    else:
        raise Exception(f'Unrecognized file format `metropolis.format`="{input_format}"')

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
