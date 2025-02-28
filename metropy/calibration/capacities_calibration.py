import os
import time

import numpy as np
import polars as pl
from sklearn.metrics import root_mean_squared_error

import metropy.utils.mpl as mpl
import metropy.utils.io as metro_io
import metropy.calibration.base as metro_calib


def read_metropolis_routes(filename: str, edges: pl.DataFrame):
    print("Reading route results...")
    lf = metro_io.scan_dataframe(filename)
    lf = lf.with_columns((pl.col("exit_time") - pl.col("entry_time")).alias("travel_time"))
    edges = edges.select(
        "edge_id",
        (pl.col("length") / pl.col("speed") + pl.col("constant_travel_time")).alias("ff_time"),
    )
    # Edges are shifted by -1 because the exit time of an edge depends mostly on the capacity of the
    # following edge (vehicles can exit an edge only when they are free to enter the following
    # edge).
    df = (
        lf.join(edges.lazy(), on="edge_id")
        .sort("trip_id", "entry_time")
        .select(
            "trip_id",
            pl.col("edge_id").shift(-1).over("trip_id").fill_null("edge_id"),
            "ff_time",
            (pl.col("travel_time") - pl.col("ff_time")).alias("congested_time"),
        )
        .drop_nulls()
        .collect()
    )
    return df


def compute_congestion_index_by_edge_characteristics(
    edges_variables: pl.DataFrame,
    route_df: pl.DataFrame,
    config_variables: dict,
    explanatory_variables: list[str],
    interaction_variables: list[list[str]],
):
    print("Computing congestion index by edge characteristics...")
    # Retrieve the congestion time and ff time for all edges in the path of the TomTom requests.
    df = route_df.group_by(pl.col("trip_id").alias("id")).agg(
        pl.col("edge_id"),
        pl.col("congested_time"),
        pl.col("ff_time"),
    )
    output = []
    # Add total congestion index for each TomTom request.
    output.append(
        df.select(
            (pl.col("congested_time").list.sum() / pl.col("ff_time").list.sum())
            .fill_nan(0.0)
            .alias("constant")
        )
    )
    for var in explanatory_variables:
        print(f"\t{var}...")
        output.append(
            df.with_columns(
                pl.col("edge_id")
                .list.eval(
                    pl.element().replace_strict(edges_variables["edge_id"], edges_variables[var])
                )
                .alias(var),
            )
            .lazy()
            .explode("congested_time", "ff_time", var)
            .group_by("id", maintain_order=True)
            .agg(
                (
                    (pl.col("congested_time") * (pl.col(var) == mod)).sum()
                    / (pl.col("ff_time") * (pl.col(var) == mod)).sum()
                )
                .fill_nan(0.0)
                .alias(f"{var}_{mod}")
                for mod in config_variables[var][1:]
            )
            .drop("id")
            .collect()
        )
    for var1, var2 in interaction_variables:
        print(f"\t{var1} x {var2}...")
        output.append(
            df.with_columns(
                pl.col("edge_id")
                .list.eval(
                    pl.element().replace_strict(edges_variables["edge_id"], edges_variables[var1])
                )
                .alias(var1),
                pl.col("edge_id")
                .list.eval(
                    pl.element().replace_strict(edges_variables["edge_id"], edges_variables[var2])
                )
                .alias(var2),
            )
            .lazy()
            .explode("congested_time", "ff_time", var1, var2)
            .group_by("id", maintain_order=True)
            .agg(
                (
                    pl.col("congested_time")
                    .filter(pl.col(var1).eq(mod1), pl.col(var2).eq(mod2))
                    .sum()
                    / pl.col("ff_time").filter(pl.col(var1).eq(mod1), pl.col(var2).eq(mod2)).sum()
                )
                .fill_nan(0.0)
                .alias(f"{var1}_{mod1}_{var2}_{mod2}")
                for mod1 in config_variables[var1][1:]
                for mod2 in config_variables[var2][1:]
            )
            .drop("id")
            .collect()
        )
    df = pl.concat(output, how="horizontal")
    return df


def get_new_capacities(
    edges_variables: pl.DataFrame,
    old_capacities: pl.DataFrame,
    explanatory_variables: list[str],
    interaction_variables: list[list[str]],
    coefs: dict[str, float],
    config_variables: dict,
    max_capacity: float,
):
    print("Computing edge-level penalties...")
    penalties = pl.repeat(coefs["constant"], len(edges_variables), eager=True).rename("penalty")
    for var in explanatory_variables:
        for mod in config_variables[var][1:]:
            coef = coefs[f"{var}_{mod}"]
            penalties += float(coef) * (edges_variables[var] == mod)
    for var1, var2 in interaction_variables:
        for mod1 in config_variables[var1][1:]:
            for mod2 in config_variables[var2][1:]:
                coef = coefs[f"{var1}_{mod1}_{var2}_{mod2}"]
                penalties += float(coef) * (
                    (edges_variables[var1] == mod1) & (edges_variables[var2] == mod2)
                )
    penalties = penalties.clip(lower_bound=0.0)
    new_df = pl.DataFrame([edges_variables["edge_id"], penalties])
    df = old_capacities.select("edge_id", "capacity").join(new_df, on="edge_id")
    df = df.with_columns(
        capacity=(pl.col("capacity") / pl.col("penalty")).clip(upper_bound=max_capacity)
    )
    return df


def plot_graphs(Y_tomtom: np.ndarray, Y_metro: np.ndarray, graph_dir: str):
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    corr = np.corrcoef(Y_tomtom, Y_metro)[0][1]
    print(f"Correlation between TomTom and calibrated values: {corr:.4%}")
    rmse = root_mean_squared_error(Y_tomtom, Y_metro)
    print(f"RMSE between TomTom and calibrated values: {rmse:.2f}")
    # Distribution of TomTom congested times (from TomTom, from simulation and predicted).
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(Y_tomtom.max(), Y_metro.max()) / 60
    bins = np.arange(0.0, m + 1.0, 1.0)
    F_tomtom, _, _ = ax.hist(
        Y_tomtom / 60,
        bins=list(bins),
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(0),
        label="TomTom",
    )
    F_metro, _, _ = ax.hist(
        Y_metro / 60,
        bins=list(bins),
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(1),
        label="METROPOLIS2",
    )
    # bin index at which all cumulative densities are over 99%.
    idx = max(
        np.searchsorted(F_tomtom, 0.99),
        np.searchsorted(F_metro, 0.99),
    )
    ax.set_xlim(0, bins[idx])
    ax.set_xlabel("Travel time (min.)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative density")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "congested_times_hist.pdf"))
    # Scatter plot of congested times (from TomTom and calibrated).
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(Y_tomtom.max(), Y_metro.max()) / 60
    ax.scatter(Y_tomtom / 60, Y_metro / 60, color=mpl.CMP(0), marker=".", s=0.1, alpha=0.1)
    ax.plot([0, m], [0, m], color="red", linewidth=0.5)
    ax.set_xlim(0, m)
    ax.set_xlabel("TomTom travel time (min.)")
    ax.set_ylim(0, m)
    ax.set_ylabel("METROPOLIS2 travel time (min.)")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "congested_times_scatter.png"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "capacities_filename",
        "run.tomtom_routes.directory",
        "calibration.map_matching.output_filename",
        "calibration.variables",
        "calibration.capacities_calibration.explanatory_variables",
        "calibration.capacities_calibration.interaction_variables",
        "calibration.capacities_calibration.maximum_capacity",
    ]
    check_keys(config, mandatory_keys)

    run_directory = config["run"]["tomtom_routes"]["directory"]

    t0 = time.time()

    edges_input = metro_io.read_dataframe(os.path.join(run_directory, "input", "edges.parquet"))

    route_df = read_metropolis_routes(
        os.path.join(run_directory, "output", "route_results.parquet"),
        edges_input,
    )

    tomtom_df = metro_calib.read_tomtom_paths(
        config["calibration"]["tomtom"]["output_filename"],
        config["calibration"]["map_matching"]["output_filename"],
        route_df,
    )

    edges_charac = metro_io.read_dataframe(config["clean_edges_file"])

    explanatory_variables = config["calibration"]["capacities_calibration"]["explanatory_variables"]
    interaction_variables = config["calibration"]["capacities_calibration"]["interaction_variables"]
    used_variables = metro_calib.get_all_used_variables(
        (explanatory_variables,), (interaction_variables,)
    )

    edges_variables, config_variables = metro_calib.create_variables_and_modalities(
        edges_charac, config["calibration"]["variables"], used_variables
    )

    exog_variables = compute_congestion_index_by_edge_characteristics(
        edges_variables,
        route_df,
        config_variables,
        explanatory_variables,
        interaction_variables,
    )

    endog_variable = tomtom_df["congested_time"] / tomtom_df["ff_time"]

    (
        _,
        residuals,
        rmse,
        coefs,
    ) = metro_calib.compute_lasso(
        endog_variable,
        exog_variables,
        config["calibration"]["capacities_calibration"],
    )
    print(coefs)

    # TODO: Write the code to adjust the capacities from the results of the LASSO.
    #  old_capacities = metro_io.read_dataframe(config["capacities_filename"])

    #  capacities = get_new_capacities(
        #  edges_variables,
        #  old_capacities,
        #  explanatory_variables,
        #  interaction_variables,
        #  coefs,
        #  config_variables,
        #  config["calibration"]["capacities_calibration"]["maximum_capacity"],
    #  )

    #  metro_io.save_dataframe(capacities, config["capacities_filename"])

    graph_dir = os.path.join(config["graph_directory"], "calibration.capacities_calibration")
    plot_graphs(
        (tomtom_df["congested_time"] + tomtom_df["ff_time"]).to_numpy(),
        route_df.group_by("trip_id")
        .agg((pl.col("congested_time").sum() + pl.col("ff_time").sum()).alias("tt"))["tt"]
        .to_numpy(),
        graph_dir,
    )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
