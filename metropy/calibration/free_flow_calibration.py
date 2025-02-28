import time

import polars as pl

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
        pl.col("path").alias("edge_id"),
        pl.col("path")
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


def save_coefs(coefs: dict[str, float], filename: str):
    print("Saving regression coefficients...")
    coef_df = pl.DataFrame({"name": coefs.keys(), "value": coefs.values()})
    metro_io.save_dataframe(coef_df, filename)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "graph_directory",
        "calibration.map_matching.output_filename",
        "calibration.variables",
        "calibration.free_flow_calibration.additive_variables",
        "calibration.free_flow_calibration.additive_interaction_variables",
        "calibration.free_flow_calibration.multiplicative_variables",
        "calibration.free_flow_calibration.multiplicative_interaction_variables",
        "calibration.free_flow_calibration.coef_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    tomtom_df = metro_calib.read_tomtom_paths(
        config["calibration"]["tomtom"]["output_filename"],
        config["calibration"]["map_matching"]["output_filename"],
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

    save_coefs(coefs, config["calibration"]["free_flow_calibration"]["coef_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
