import time

import numpy as np
import polars as pl

import metropy.utils.io as metro_io
import metropy.emissions.base as metro_emissions

# Emitter height (in m)
H = 0.5
# Receptor height (in m)
Z = 1.5
# Wind speed (in m/sec)
WIND_SPEED = 15 / 3.6
# Wind angle (in radian, relative to positive x-axis).
# Note: use `np.deg2rad` to convert from degree to radian.
# W->E: 0 degrees.
# S->N: 90 degrees.
# E->W: 180 degrees.
# N->S: 270 degrees.
WIND_DIR = np.deg2rad(0)

# Set the parameter of the plume
A = 0.0787
B = 0.0014
C = 0.135
D = 0.0475
E = 0.465


def read_emissions(filename: str):
    print("Reading emissions...")
    return metro_io.read_dataframe(filename)


def merge_with_emissions(
    gridded_edges: pl.DataFrame,
    emissions: pl.DataFrame,
    pollutants: list[str],
    simulation_ratio: float,
):
    df = (
        gridded_edges.lazy()
        .join(emissions.lazy(), on="edge_id")
        .with_columns(pl.col(p) * pl.col("length_share") for p in pollutants)
        .group_by("x0", "y0", "time_period")
        .agg(pl.col("cell_id").first(), *(pl.col(p).sum() / simulation_ratio for p in pollutants))
        .sort("time_period", "x0", "y0")
        .collect()
    )
    return df


def gaussian_plume_model(
    gridded_emissions: pl.DataFrame,
    grid: pl.DataFrame,
    pollutants: list[str],
    resolution: float,
):
    dfs = list()
    half_resolution = resolution / 2
    for (time_period,), df in gridded_emissions.partition_by(
        "time_period", maintain_order=True, as_dict=True, include_key=False
    ).items():
        print(f"Time period: {time_period}")
        lf = (
            df.lazy()
            .rename({"cell_id": "cell_id_emitter", "x0": "x_emitter", "y0": "y_emitter"})
            .join(grid.lazy().rename({"cell_id": "cell_id_receptor", "x0": "x_receptor", "y0": "y_receptor"}), how="cross")
        )
        # Compute distance from emitter to receptor in the x- and y-axis.
        lf = lf.with_columns(
            dx=pl.col("x_receptor") - pl.col("x_emitter"),
            dy=pl.col("y_receptor") - pl.col("y_emitter"),
        )
        # The distance is offset by `half_resolution` so that the receptors are not perfectly
        # aligned with the emitters.
        lf = lf.with_columns(
            dx=pl.col("dx") + (pl.col("dx") >= 0) * half_resolution,
            dy=pl.col("dy") + (pl.col("dy") >= 0) * half_resolution,
        )
        # Compute the angle of the emitter-to-receptor line (in radian).
        lf = lf.with_columns(pl.arctan2("dy", "dx").alias("angle"))
        # Compute the absolute angle difference (in radian) between wind and emitter-to-receptor line.
        # The values range between 0 and 2 pi, where 0 is same angle, -pi is opposite angle.
        lf = lf.with_columns(((pl.lit(WIND_DIR) - pl.col("angle")) % (2 * np.pi)).alias("dangle"))
        # Filter out receptors opposite of the wind direction.
        lf = lf.filter(~pl.col("dangle").is_between(np.pi / 2, 3 * np.pi / 2))
        # Compute the distance from emitter to receptor.
        lf = lf.with_columns((pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt().alias("dist"))
        # Compute distance parallel (x) and perpendicular (y) to the wind.
        lf = lf.with_columns(
            (pl.col("dangle").cos() * pl.col("dist")).alias("x"),
            (pl.col("dangle").sin() * pl.col("dist")).alias("y"),
        )
        # Filter out receptors with null parallel distance.
        lf = lf.filter(pl.col("x") > 0)
        # Compute sigma_y and sigma_z.
        lf = lf.with_columns(
            (A * pl.col("x") / (1 + B * pl.col("x")) ** C).alias("sigma_y"),
            (D * pl.col("x") / (1 + B * pl.col("x")) ** E).alias("sigma_z"),
        )
        # Compute the dispersion factor (Gaussian plume equation).
        lf = lf.with_columns(
            (
                (-pl.col("y") ** 2 / (2 * pl.col("sigma_y") ** 2)).exp()
                * (
                    (-((Z + H) ** 2) / (2 * pl.col("sigma_z") ** 2)).exp()
                    + (-((Z - H) ** 2) / (2 * pl.col("sigma_z") ** 2)).exp()
                )
                / (2 * np.pi * WIND_SPEED * pl.col("sigma_y") * pl.col("sigma_z"))
            ).alias("factor")
        )
        # Multiply the factor by the emissions at the emitter (converted from g/h to μg/s).
        lf = lf.with_columns([(pl.col(p) * 1e6 / 3600) * pl.col("factor") for p in pollutants])
        # Sum the pollutants by receptor.
        lf = lf.group_by("cell_id_receptor").agg([pl.col(p).sum() for p in pollutants])
        # Remove the receptors without any pollution.
        lf = lf.filter(pl.any_horizontal((pl.col(p) > 0 for p in pollutants)))
        # Add the receptor's x / y again.
        lf = lf.rename({"cell_id_receptor": "cell_id"}).join(grid.lazy(), on="cell_id").select(
            "cell_id", "x0", "y0", *pollutants, time_period=pl.lit(time_period, dtype=pl.UInt8)
        )
        lf = lf.sort("x0", "y0")
        # Collect the resulting DataFrame in `streaming` mode to reduce memory use.
        df_results = lf.collect(streaming=True)
        dfs.append(df_results)
    df = pl.concat(dfs, how="vertical")
    df = chemical_model(df)
    print(
        df.select(
            col for col in df.columns if col not in ("x0", "y0", "cell_id", "time_period")
        ).describe()
    )
    return df


def chemical_model(df: pl.DataFrame):
    if "NOx" in df.columns:
        df = df.with_columns(
            ((pl.col("NOx") * 103) / (pl.col("NOx") + 103) + .0005 * pl.col("NOx")).alias("NO2")
        ).with_columns(
            (pl.col("NOx") - pl.col("NO2")).alias("NO")
        )
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "crs",
        "emisens.edge_emissions",
        "emisens.pollutants",
        "metro-trace.grid_resolution",
        "metro-trace.gridded_emissions_filename",
        "metro-trace.concentration_filename",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["emisens"]

    t0 = time.time()

    edges = metro_emissions.read_edge_length(config["clean_edges_file"], config["crs"])

    grid_gdf, grid = metro_emissions.create_grid(
        edges, config["metro-trace"]["grid_resolution"], config["crs"]
    )

    gridded_edges = metro_emissions.get_gridded_edges(edges, grid_gdf)

    pollutants = config["emisens"]["pollutants"]
    if "EC" in pollutants:
        # Energy consumption cannot be dispersed.
        pollutants.remove("EC")
        # TODO. I should add a config option "compute_EC" instead.

    emissions = read_emissions(config["emisens"]["edge_emissions"])

    gridded_emissions = merge_with_emissions(
        gridded_edges, emissions, pollutants, config.get("run", dict).get("simulation_ratio", 1.0)
    )

    metro_io.save_dataframe(gridded_emissions, config["metro-trace"]["gridded_emissions_filename"])

    concentrations = gaussian_plume_model(
        gridded_emissions, grid, pollutants, config["metro-trace"]["grid_resolution"]
    )

    metro_io.save_dataframe(concentrations, config["metro-trace"]["concentration_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
