# Base functions used by multiple scripts to generate simulation input.
import polars as pl

import metropy.utils.io as metro_io


def main(edge_filename: str, default_capacities: dict[str, float]):
    print("Reading edges")
    lf = metro_io.scan_dataframe(edge_filename)
    lf = lf.select(
        "edge_id",
        capacity=pl.col("road_type").replace_strict(default_capacities, return_dtype=pl.Float64),
    )
    return lf.collect()


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "default_edge_capacities",
        "capacities_filename",
    ]
    check_keys(config, mandatory_keys)

    df = main(config["clean_edges_file"], config["default_edge_capacities"])
    metro_io.save_dataframe(df, config["capacities_filename"])
