import os

import polars as pl

import metropy.utils.io as metro_io


def read_route_results(filename: str, edge_id: str):
    print("Reading route results...")
    lf = metro_io.scan_dataframe(filename)
    # TODO: Manage various PCEs.
    # Select agents who take the edge as part of their trips.
    lf = lf.filter(pl.col("edge_id").eq(edge_id).any().over("agent_id"))
    n = lf.select(pl.col("edge_id").eq(edge_id).sum()).collect().item()
    print(f"Number of unique trips taking edge {edge_id}: {n:,}")
    return lf


def get_flows(lf: pl.LazyFrame):
    print("Computing flows...")
    edge_counts = lf.group_by("edge_id").len().collect()
    return edge_counts


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "run_directory",
        "road_results.chevelus.edge_id",
        "road_results.chevelus.output_filename",
    ]
    check_keys(config, mandatory_keys)

    lf = read_route_results(
        os.path.join(config["run_directory"], "output", "route_results.parquet"),
        config["road_results"]["chevelus"]["edge_id"],
    )
    edge_counts = get_flows(lf)
    print("Saving output file...")
    metro_io.save_dataframe(edge_counts, config["road_results"]["chevelus"]["output_filename"])
