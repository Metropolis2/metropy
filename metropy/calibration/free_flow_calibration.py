import os
import time

import numpy as np
import polars as pl
from sklearn.metrics import root_mean_squared_error

import metropy.utils.mpl as mpl
import metropy.utils.io as metro_io
import metropy.calibration.base as metro_calib


def compute_regression_variables(
    edges_variables: pl.DataFrame,
    edges_charac: pl.DataFrame,
    tomtom_df: pl.DataFrame,
    config_variables: dict,
    additive_variables: list[str],
    additive_interaction_variables: list[list[str]],
    multiplicative_variables: list[str],
    multiplicative_interaction_variables: list[list[str]],
):
    print("Computing congested time by edge characteristics...")
    # Edge-level base free-flow travel time (in seconds).
    ff_tts = 3.6 * edges_charac["length"] / edges_charac["speed_limit"]
    df = tomtom_df.select(
        "id",
        pl.col("cpath").alias("edge_id"),
        pl.col("cpath")
        .list.eval(pl.element().replace_strict(edges_charac["edge_id"], ff_tts))
        .alias("ff_tt"),
    )
    output = []
    output.append(
        df.select(
            pl.col("edge_id").list.len().alias("nb_edges"),
            pl.col("ff_tt").list.sum().alias("base_ff_tt"),
        )
    )
    print("Additive variables...")
    for var in additive_variables:
        print(f"\t{var}...")
        # `tomtom_var_data` is a pl.Series where each element is a list with the value of `var`
        # for each edge on the path. The elements are ordered by TomTom request id.
        tomtom_var_data = df["edge_id"].list.eval(
            pl.element().replace_strict(edges_variables["edge_id"], edges_variables[var])
        )
        results = list()
        for mod in config_variables[var][1:]:
            values = tomtom_var_data.list.count_matches(mod)
            results.append(pl.Series(values=values, name=f"{var}_{mod}"))
        output.append(pl.DataFrame(results))
    print("Additive interaction variables...")
    for var1, var2 in additive_interaction_variables:
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
            .explode(var1, var2)
            .group_by("id", maintain_order=True)
            .agg(
                pl.col("id")
                .filter(pl.col(var1).eq(mod1), pl.col(var2).eq(mod2))
                .count()
                .alias(f"{var1}_{mod1}_{var2}_{mod2}")
                for mod1 in config_variables[var1][1:]
                for mod2 in config_variables[var2][1:]
            )
            .drop("id")
            .collect()
        )
    print("Multiplicative variables...")
    for var in multiplicative_variables:
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
            .explode("ff_tt", var)
            .group_by("id", maintain_order=True)
            .agg(
                (pl.col("ff_tt") * (pl.col(var) == mod)).sum().alias(f"tt_{var}_{mod}")
                for mod in config_variables[var][1:]
            )
            .drop("id")
            .collect()
        )
    print("Multiplicative interaction variables...")
    for var1, var2 in multiplicative_interaction_variables:
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
            .explode("ff_tt", var1, var2)
            .group_by("id", maintain_order=True)
            .agg(
                pl.col("ff_tt")
                .filter(pl.col(var1).eq(mod1), pl.col(var2).eq(mod2))
                .sum()
                .alias(f"tt_{var1}_{mod1}_{var2}_{mod2}")
                for mod1 in config_variables[var1][1:]
                for mod2 in config_variables[var2][1:]
            )
            .drop("id")
            .collect()
        )
    df = pl.concat(output, how="horizontal")
    return df


def get_penalties(
    edges_variables: pl.DataFrame,
    additive_variables: list[str],
    additive_interaction_variables: list[list[str]],
    multiplicative_variables: list[str],
    multiplicative_interaction_variables: list[list[str]],
    coefs: dict[str, float],
    config_variables: dict,
):
    print("Computing edge-level penalties...")
    additive_penalties = get_penalties_inner(
        edges_variables,
        additive_variables,
        additive_interaction_variables,
        coefs,
        "additive_penalty",
        "nb_edges",
        "",
        config_variables,
    )
    multiplicative_penalties = 1 / get_penalties_inner(
        edges_variables,
        multiplicative_variables,
        multiplicative_interaction_variables,
        coefs,
        "multiplicative_penalty",
        "base_ff_tt",
        "tt_",
        config_variables,
    )
    penalties = pl.DataFrame(
        [edges_variables["edge_id"], additive_penalties, multiplicative_penalties]
    )
    return penalties


def get_penalties_inner(
    edges_variables: pl.DataFrame,
    variables: list[str],
    interaction_variables: list[list[str]],
    coefs: dict[str, float],
    name: str,
    cst_var: str,
    prefix: str,
    config_variables: dict,
):
    values = pl.repeat(coefs[cst_var], len(edges_variables), eager=True).rename(name)
    for var in variables:
        for mod in config_variables[var][1:]:
            coef = coefs[f"{prefix}{var}_{mod}"]
            values += float(coef) * (edges_variables[var] == mod)
    for var1, var2 in interaction_variables:
        for mod1 in config_variables[var1][1:]:
            for mod2 in config_variables[var2][1:]:
                coef = coefs[f"{prefix}{var1}_{mod1}_{var2}_{mod2}"]
                values += float(coef) * (
                    (edges_variables[var1] == mod1) & (edges_variables[var2] == mod2)
                )
    return values


def apply_penalty_bounds(penalties: pl.DataFrame, edges_charac: pl.DataFrame, config: dict):
    print("Applying penalty constraints...")
    # Apply the constraint on the additive penalty.
    additive_lb = float(config.get("additive_penalty_lower_bound", 0.0))
    n = (penalties["additive_penalty"] < additive_lb).sum()
    if n:
        print(
            "{:,} edges ({:.2%}) have an additive penalty below the lower bound".format(
                n, n / len(penalties)
            )
        )
        penalties = penalties.with_columns(pl.col("additive_penalty").clip(lower_bound=additive_lb))
    # Apply the constraint on the multiplicative penalty.
    assert (edges_charac["edge_id"] == penalties["edge_id"]).all()
    speed_lb = config.get("road_speed_lower_bound", 1.0)
    penalties = penalties.with_columns(
        (edges_charac["speed_limit"] * pl.col("multiplicative_penalty")).alias("speed")
    )
    n = (penalties["speed"] < speed_lb).sum()
    if n:
        print(
            "{:,} edges ({:.2%}) have a road speed below the lower bound".format(
                n, n / len(penalties)
            )
        )
        penalties = penalties.with_columns(pl.col("speed").clip(lower_bound=speed_lb))
    ub = config.get("global_speed_upper_bound")
    if ub is None:
        return penalties
    # Minimum edge travel time so that the constraint is satisfied.
    df = pl.DataFrame({"min_tot_tt": edges_charac["length"] / (ub / 3.6)})
    # Maximum edge speed to satisfy the constraint.
    df = df.with_columns(
        max_speed=pl.when(pl.col("min_tot_tt") > penalties["additive_penalty"])
        .then(3.6 * edges_charac["length"] / (pl.col("min_tot_tt") - penalties["additive_penalty"]))
        .otherwise(pl.lit(np.inf))
    )
    n = (df["max_speed"] < penalties["speed"]).sum()
    if n:
        print(
            "{:,} edges ({:.2%}) have a global speed above the upper bound".format(
                n, n / len(penalties)
            )
        )
        penalties = penalties.with_columns(
            pl.min_horizontal("speed", df["max_speed"]).alias("speed")
        )
    # Add edge travel time.
    penalties = penalties.with_columns(
        (pl.col("additive_penalty") + edges_charac["length"] / (pl.col("speed") / 3.6)).alias(
            "travel_time"
        )
    )
    return penalties


def plot_graphs(Y_tomtom: np.ndarray, Y_base: np.ndarray, Y_calib: np.ndarray, graph_dir: str):
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    corr = np.corrcoef(Y_tomtom, Y_base)[0][1]
    print(f"Correlation between TomTom and uncalibrated values: {corr:.4%}")
    corr = np.corrcoef(Y_tomtom, Y_calib)[0][1]
    print(f"Correlation between TomTom and calibrated values: {corr:.4%}")
    rmse = root_mean_squared_error(Y_tomtom, Y_base)
    print(f"RMSE between TomTom and uncalibrated values: {rmse:.2f}")
    rmse = root_mean_squared_error(Y_tomtom, Y_calib)
    print(f"RMSE between TomTom and calibrated values: {rmse:.2f}")
    # Distribution of free-flow travel times (from TomTom, uncalibrated and calibrated).
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(Y_tomtom.max(), Y_base.max(), Y_calib.max()) / 60
    bins = np.arange(0.0, m + 1.0, 2.0)
    F_tomtom, _, _ = ax.hist(
        Y_tomtom / 60,
        bins=list(bins),
        density=True,
        #  cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(0),
        label="TomTom",
    )
    F_metro, _, _ = ax.hist(
        Y_base / 60,
        bins=list(bins),
        density=True,
        #  cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(1),
        label="Uncalibrated",
    )
    F_hat, _, _ = ax.hist(
        Y_calib / 60,
        bins=list(bins),
        density=True,
        #  cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(2),
        label="Calibrated",
    )
    # bin index at which all cumulative densities are over 99%.
    idx = max(
        np.searchsorted(F_tomtom, 0.99),
        np.searchsorted(F_metro, 0.99),
        np.searchsorted(F_hat, 0.99),
    )
    ub = bins[idx]
    ax.set_xlim(0, ub)
    ax.set_xlabel("Free-flow travel time (min.)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "ff_times_hist.pdf"))
    # Scatter plot of free-flow travel times (from TomTom and calibrated).
    fig, ax = mpl.get_figure(fraction=0.8)
    m = max(Y_tomtom.max(), Y_calib.max()) / 60
    ax.scatter(Y_tomtom / 60, Y_calib / 60, color=mpl.CMP(0), marker=".", s=0.1, alpha=0.1)
    ax.plot([0, m], [0, m], color="red", linewidth=0.5)
    ax.set_xlim(0, ub)
    ax.set_xlabel("TomTom free-flow travel time (min.)")
    ax.set_ylim(0, ub)
    ax.set_ylabel("Calibrated free-flow travel time (min.)")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "ff_times_scatter.png"))
    # Distribution of residuals.
    fig, ax = mpl.get_figure(fraction=0.8)
    residuals = (Y_tomtom - Y_calib) / 60
    ax.hist(
        residuals,
        bins=120,
        density=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(0),
    )
    lb, rb = np.quantile(residuals, [0.001, 0.999])
    m = max(abs(lb), abs(rb))
    ax.set_xlim(-m, m)
    ax.set_xlabel("Residuals (TomTom $-$ calibrated, min.)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Density")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "residuals_hist.pdf"))


def save_dataframes(penalties: pl.DataFrame, coefs: dict[str, float], config: dict):
    print("Saving penalties...")
    metro_io.save_dataframe(penalties, config["output_filename"])
    if "coef_filename" in config:
        print("Saving regression coefficients...")
        coef_df = pl.DataFrame({"name": coefs.keys(), "value": coefs.values()})
        metro_io.save_dataframe(coef_df, config["coef_filename"])


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "graph_directory",
        "calibration.post_map_matching.output_filename",
        "calibration.variables",
        "calibration.free_flow_calibration.additive_variables",
        "calibration.free_flow_calibration.additive_interaction_variables",
        "calibration.free_flow_calibration.multiplicative_variables",
        "calibration.free_flow_calibration.multiplicative_interaction_variables",
        "calibration.free_flow_calibration.output_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    tomtom_df = metro_calib.read_tomtom_paths(
        config["calibration"]["post_map_matching"]["output_filename"]
    )

    edges_charac = metro_io.read_dataframe(config["clean_edges_file"])

    additive_variables = config["calibration"]["free_flow_calibration"]["additive_variables"]
    additive_interaction_variables = config["calibration"]["free_flow_calibration"][
        "additive_interaction_variables"
    ]
    multiplicative_variables = config["calibration"]["free_flow_calibration"][
        "multiplicative_variables"
    ]
    multiplicative_interaction_variables = config["calibration"]["free_flow_calibration"][
        "multiplicative_interaction_variables"
    ]
    used_variables = metro_calib.get_all_used_variables(
        (additive_variables, multiplicative_variables),
        (additive_interaction_variables, multiplicative_interaction_variables),
    )

    edges_variables, config_variables = metro_calib.create_variables_and_modalities(
        edges_charac, config["calibration"]["variables"], used_variables
    )

    exog_variables = compute_regression_variables(
        edges_variables,
        edges_charac,
        tomtom_df,
        config_variables,
        additive_variables,
        additive_interaction_variables,
        multiplicative_variables,
        multiplicative_interaction_variables,
    )

    endog_variable = tomtom_df["ff_time"]

    (
        Y_hat,
        residuals,
        rmse,
        coefs,
    ) = metro_calib.compute_lasso(
        endog_variable,
        exog_variables,
        config["calibration"]["free_flow_calibration"],
    )

    penalties = get_penalties(
        edges_variables,
        additive_variables,
        additive_interaction_variables,
        multiplicative_variables,
        multiplicative_interaction_variables,
        coefs,
        config_variables,
    )

    penalties = apply_penalty_bounds(
        penalties, edges_charac, config["calibration"]["free_flow_calibration"]
    )

    Y_calib = (
        tomtom_df["cpath"]
        .list.eval(pl.element().replace_strict(penalties["edge_id"], penalties["travel_time"]))
        .list.sum()
        .to_numpy()
    )

    graph_dir = os.path.join(config["graph_directory"], "calibration.free_flow_calibration")
    plot_graphs(
        endog_variable.to_numpy(), exog_variables["base_ff_tt"].to_numpy(), Y_calib, graph_dir
    )

    save_dataframes(penalties, coefs, config["calibration"]["free_flow_calibration"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
