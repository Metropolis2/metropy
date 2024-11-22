import os
import time

import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io
import metropy.emissions.base as metro_emissions


def read_dataframes(dir1: str, dir2: str, filename: str, columns: list[str], suffix: str):
    df1 = pl.read_parquet(
        os.path.join(dir1, filename), columns=["cell_id", "time_period", *columns]
    )
    df2 = pl.read_parquet(
        os.path.join(dir2, filename), columns=["cell_id", "time_period", *columns]
    )
    df = (
        df1.join(df2, on=["cell_id", "time_period"], how="full", coalesce=True)
        .select(
            "cell_id",
            "time_period",
            *sum(
                (
                    [
                        pl.col(col).fill_null(0.0).alias(f"{col}_{suffix}1"),
                        pl.col(f"{col}_right").fill_null(0.0).alias(f"{col}_{suffix}2"),
                    ]
                    for col in columns
                ),
                start=[],
            ),
        )
        .with_columns(
            (pl.col(f"{col}_{suffix}2") - pl.col(f"{col}_{suffix}1")).alias(f"{col}_{suffix}_diff")
            for col in columns
        )
    )
    return df


def read_emissions(dir1: str, dir2: str, filename: str, pollutants: list[str]):
    print("Reading emissions...")
    return read_dataframes(dir1, dir2, filename, pollutants, "emission")


def read_concentrations(dir1: str, dir2: str, filename: str, pollutants: list[str]):
    print("Reading concentrations...")
    if "NOx" in pollutants:
        pollutants.extend(["NO", "NO2"])
    return read_dataframes(dir1, dir2, filename, pollutants, "concentration")


def read_population(dir1: str, dir2: str, filename: str):
    print("Reading population dispersion...")
    df1 = (
        pl.scan_parquet(os.path.join(dir1, filename))
        .group_by("cell_id", "time_period")
        .agg(pl.col("duration_h").sum())
        .collect()
    )
    df2 = (
        pl.scan_parquet(os.path.join(dir2, filename))
        .group_by("cell_id", "time_period")
        .agg(pl.col("duration_h").sum())
        .collect()
    )
    df = (
        df1.join(df2, on=["cell_id", "time_period"], how="full", suffix="2", coalesce=True)
        .rename({"duration_h": "duration_h1"})
        .with_columns(
            pl.col("duration_h1").fill_null(0.0),
            pl.col("duration_h2").fill_null(0.0),
        )
        .with_columns((pl.col("duration_h2") - pl.col("duration_h1")).alias("duration_h_diff"))
    )
    return df


def read_exposure(dir1: str, dir2: str, filename: str, pollutants: list[str]):
    print("Reading exposure...")
    df1 = (
        pl.scan_parquet(os.path.join(dir1, filename))
        .group_by("cell_id", "time_period")
        .agg(pl.col(p).sum() for p in pollutants)
        .collect()
    )
    df2 = (
        pl.scan_parquet(os.path.join(dir2, filename))
        .group_by("cell_id", "time_period")
        .agg(pl.col(p).sum() for p in pollutants)
        .collect()
    )
    df = (
        df1.join(df2, on=["cell_id", "time_period"], how="full", coalesce=True)
        .select(
            "cell_id",
            "time_period",
            *sum(
                (
                    [
                        pl.col(p).fill_null(0.0).alias(f"{p}_exposure1"),
                        pl.col(f"{p}_right").fill_null(0.0).alias(f"{p}_exposure2"),
                    ]
                    for p in pollutants
                ),
                start=[],
            ),
        )
        .with_columns(
            (pl.col(f"{p}_exposure2") - pl.col(f"{p}_exposure1")).alias(f"{p}_exposure_diff")
            for p in pollutants
        )
    )
    return df


def merge(
    df_emissions: pl.DataFrame,
    df_concentration: pl.DataFrame,
    df_population: pl.DataFrame,
    df_exposure: pl.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
):
    print("Merging data...")
    df = (
        df_emissions.join(df_concentration, on=["cell_id", "time_period"], how="full", coalesce=True)
        .join(df_population, on=["cell_id", "time_period"], how="full", coalesce=True)
        .join(df_exposure, on=["cell_id", "time_period"], how="full", coalesce=True)
    )
    assert df["cell_id"].is_in(grid_gdf["cell_id"]).all()
    gdf = gpd.GeoDataFrame(grid_gdf.merge(df.to_pandas(), on="cell_id"))
    return gdf


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "crs",
        "clean_edges_file",
        "emisens.pollutants",
        "metro-trace.grid_resolution",
        "metro-trace.gridded_emissions_filename",
        "metro-trace.concentration_filename",
        "metro-trace.population_dispersion_filename",
        "metro-trace.exposure_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    dir1 = "./output/emissions/"
    dir2 = "./output/emissions_lez/"

    pollutants = config["emisens"]["pollutants"]
    if "EC" in pollutants:
        # Energy consumption cannot be dispersed.
        pollutants.remove("EC")
        # TODO. I should add a config option "compute_EC" instead.

    df_emissions = read_emissions(dir1, dir2, "gridded_emissions.parquet", pollutants)

    df_concentration = read_concentrations(dir1, dir2, "concentrations.parquet", pollutants)

    df_population = read_population(dir1, dir2, "population_dispersion.parquet")

    pollutants = list(config["metro-trace"]["exposure_probs"].keys())
    df_exposure = read_exposure(dir1, dir2, "exposure.parquet", pollutants)

    edges = metro_emissions.read_edge_length(config["clean_edges_file"], config["crs"])
    grid_gdf, _ = metro_emissions.create_grid(
        edges, config["metro-trace"]["grid_resolution"], config["crs"]
    )

    gdf = merge(df_emissions, df_concentration, df_population, df_exposure, grid_gdf)
    metro_io.save_geodataframe(gdf, os.path.join(dir1, "pollution_comparison.parquet"))

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
