import os

import polars as pl

import metropy.utils.io as metro_io
import metropy.run.base as metro_run


def read_route_results(filename: str):
    print("Reading route results...")
    lf = metro_io.scan_dataframe(filename)
    # TODO: Manage various PCEs.
    lf = lf.with_columns(
        (pl.col("exit_time") - pl.col("entry_time")).alias("travel_time"),
    )
    return lf


def get_values(lf: pl.LazyFrame, edges: pl.DataFrame):
    df = (
        lf.group_by("edge_id")
        .agg(
            pl.len().alias("flow"),
            pl.col("travel_time").max().alias("max_tt"),
            pl.col("travel_time").mean().alias("mean_tt"),
        )
        .collect()
    )
    edges = edges.select(
        "edge_id",
        (pl.col("length") / pl.col("speed") + pl.col("constant_travel_time")).alias("ff_tt"),
    )
    df = edges.join(df, on="edge_id", how="left").with_columns(
        pl.col("flow").fill_null(0),
    )
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "run_directory",
        "clean_edges_file",
        "road_results.edge_values.output_filename",
        "run",
    ]
    check_keys(config, mandatory_keys)

    lf = read_route_results(
        os.path.join(config["run_directory"], "output", "route_results.parquet"),
    )
    edges = metro_run.read_edges(
        config["clean_edges_file"],
        config.get("routing", dict).get("road_split", dict).get("main_edges_filename"),
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
        None,
    )
    edges = metro_run.generate_edges(edges, config["run"])
    df = get_values(lf, edges)
    print("Saving output file...")
    metro_io.save_dataframe(df, config["road_results"]["edge_values"]["output_filename"])
