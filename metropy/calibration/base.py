import polars as pl
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LassoCV

import metropy.utils.io as metro_io


def read_tomtom_paths(tomtom_filename: str, matching_filename: str, route_df: pl.DataFrame | None = None):
    print("Reading TomTom paths...")
    tomtom = metro_io.scan_dataframe(tomtom_filename)
    matching = metro_io.scan_dataframe(matching_filename)
    lf = tomtom.join(matching, on="id", how="inner")
    if route_df is not None:
        # Select only the TomTom requests for which the route was computed.
        lf = lf.join(
            route_df.lazy(), left_on=pl.col("id").cast(pl.UInt64), right_on="trip_id", how="semi"
        )
    df = (
        lf.sort("id")
        .select(
            "id",
            "path",
            (pl.col("tt_historical") - pl.col("tt_no_traffic"))
            .alias("congested_time")
            .cast(pl.Float64),
            pl.col("tt_no_traffic").alias("ff_time").cast(pl.Float64),
        )
        .collect()
    )
    if route_df is not None:
        assert len(df) == route_df["trip_id"].n_unique()
    print(f"Number of paths read: {len(df):,}")
    return df


def get_all_used_variables(
    base_variables: tuple[list[str], ...], interaction_variables: tuple[list[list[str]], ...]
):
    used_variables = set()
    for var in base_variables:
        if not isinstance(var, list):
            raise Exception("The variables must be given as a list of strings")
        used_variables = used_variables.union(set(var))
    for var in interaction_variables:
        try:
            var_set = set(sum(((var1, var2) for var1, var2 in var), start=()))
        except (ValueError, TypeError):
            raise Exception("The interaction variables must be given as a list of variable pairs")
        used_variables = used_variables.union(var_set)
    return used_variables


def create_variables_and_modalities(edges: pl.DataFrame, config: dict, used_variables: set[str]):
    print("Computing edge characteristics...")
    config_variables = {}
    output = {"edge_id": edges["edge_id"]}
    for var, kind in config.items():
        if var not in used_variables:
            continue
        assert var in edges.columns, f"Missing variable: `{var}`"
        if isinstance(kind, dict):
            assert (
                len(set(edges[var]).difference(kind.keys())) == 0
            ), "Not all values are specified in the table for variable `{var}`"
            data = edges[var].replace_strict(kind)
            unique_values = set(data)
            modalities = list()
            modalities_set = set()
            for mod in kind.values():
                if mod in unique_values and mod not in modalities_set:
                    modalities.append(mod)
                    modalities_set.add(mod)
        elif isinstance(kind, bool):
            assert edges[var].dtype == pl.Boolean, "Variable `{var}` is not of boolean type"
            data = edges[var]
            modalities = (False, True)
        elif isinstance(kind, list):
            assert edges[var].dtype.is_numeric(), "Variable `{var}` is not of numeric type"
            data = edges[var].cut(kind)
            modalities = tuple(sorted(set(data)))
        elif isinstance(kind, int):
            assert edges[var].dtype.is_integer(), f"Variable `{var}` is not of integer type"
            m = edges[var].min()
            M = kind + 1
            data = edges[var].clip(upper_bound=M)
            modalities = tuple(range(m, M + 1))
        else:
            raise Exception(f"Unsupported variable kind used for variable `{var}`: `{kind}`")
        output[var] = data
        config_variables[var] = modalities
    df = pl.DataFrame(output)
    return df, config_variables


def compute_lasso(
    endog_variable: pl.Series,
    exog_variables: pl.DataFrame,
    config: dict,
):
    print("Fitting a LASSO model...")
    Y = endog_variable.to_numpy()
    X = exog_variables.to_numpy()
    print(f"Number of observations: {X.shape[0]}")
    print(f"Number of variables: {X.shape[1]}")
    lassocv = LassoCV(alphas=config.get("alphas"), cv=10, precompute=True, fit_intercept=False,
                      max_iter=1_000_000)
    lassocv.fit(X, Y)
    print("Value of the penalization factor: {}".format(lassocv.alpha_))
    Y_hat = lassocv.predict(X)
    residuals = Y - Y_hat
    rmse = root_mean_squared_error(Y, Y_hat)
    print(f"RMSE: {rmse}")
    coefs = lassocv.coef_
    coef_lasso = {var: coef for var, coef in zip(exog_variables.columns, coefs)}
    return (Y_hat, residuals, rmse, coef_lasso)


