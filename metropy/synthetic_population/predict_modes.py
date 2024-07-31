import os
import time

import numpy as np
import polars as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt

import metropy.utils.io as metro_io
import metropy.utils.mpl as mpl


def read_egt(directory: str, mode_dtype: pl.Enum):
    print("Reading EGT...")
    cols = ["POIDSP", "MODP_H7", "DPORTEE", "ORH", "ORDEP", "DESTDEP"]
    df = pl.read_csv(
        os.path.join(directory, "Format_csv", "Deplacements_semaine.csv"), columns=cols
    )
    print("Number of trips imported: {:,}".format(len(df)))
    df = df.drop_nulls()
    df = (
        df.filter(pl.col("MODP_H7") != 6)
        .with_columns(
            pl.col("MODP_H7")
            .replace_strict(
                {
                    1: "public_transit",
                    2: "car_driver",
                    3: "car_passenger",
                    4: "motorcycle",
                    5: "bicycle",
                    7: "walking",
                }
            )
            .cast(mode_dtype)
            .alias("mode"),
            pl.col("ORDEP").cast(pl.String),
            pl.col("DESTDEP").cast(pl.String),
        )
        .drop("MODP_H7")
    )
    df = df.rename(
        {
            "POIDSP": "weight",
            "DPORTEE": "od_distance",
            "ORH": "departure_hour",
            "ORDEP": "departement_origin",
            "DESTDEP": "departement_destination",
        }
    )
    print("Number of valid trips: {:,}".format(len(df)))
    df = process_trips(df)
    return df


def read_population(directory: str, df_survey: pl.DataFrame):
    print("Reading synthetic population...")
    lf = pl.scan_parquet(os.path.join(directory, "trips.parquet"))
    lf_zones = pl.scan_parquet(os.path.join(directory, "trip_zones.parquet"))
    lf = lf.join(lf_zones, on="trip_id")
    lf = lf.with_columns(departure_hour=pl.col("departure_time") // 3600)
    df = lf.select(
        "trip_id", "od_distance", "departure_hour", "departement_origin", "departement_destination"
    ).collect()
    print("Number of trips: {:,}".format(len(df)))
    # Re-normalize od_distance to match travel survey mean.
    pop_mean = df["od_distance"].mean()
    survey_mean = (df_survey["od_distance"] * df_survey["weight"]).sum() / df_survey["weight"].sum()
    df = df.with_columns(pl.col("od_distance") * survey_mean / pop_mean)
    # TODO: This is specific to the Ile-de-France case and should be handled differently.
    idf_dep_dict = {"27": "78", "45": "91", "60": "95"}
    df = df.with_columns(
        pl.col("departement_origin").replace(idf_dep_dict),
        pl.col("departement_destination").replace(idf_dep_dict),
    )
    df = process_trips(df)
    return df


def process_trips(df: pl.DataFrame):
    df = df.with_columns(
        pl.col("od_distance").pow(2).alias("od_distance_squared"),
        pl.col("departement_origin")
        .eq(pl.col("departement_destination"))
        .alias("same_departement"),
        pl.col("departure_hour").is_between(6, 9).alias("morning_departure"),
        pl.col("departure_hour").is_between(16, 19).alias("evening_departure"),
    )
    return df


def get_exog_variables(df: pl.DataFrame, first_dep):
    X = pl.concat(
        (
            df.select(
                pl.lit(1).alias("const"),
                "od_distance",
                "od_distance_squared",
                "same_departement",
                "morning_departure",
                "evening_departure",
            ),
            df["departement_origin"].to_dummies().drop(f"departement_origin_{first_dep}"),
            df["departement_destination"].to_dummies().drop(f"departement_destination_{first_dep}"),
        ),
        how="horizontal",
    )
    return X


def estimate_model(df_survey: pl.DataFrame, mode_dtype: pl.Enum, first_dep):
    print("Estimating a Multinomial Logit model...")
    X = get_exog_variables(df_survey, first_dep)
    Y = df_survey["mode"].to_physical().to_numpy()
    model = sm.MNLogit(Y, X.to_numpy())
    results = model.fit(method="newton", disp=False)
    print(results.summary(xname=X.columns))
    print("Predicting probabilities on the travel survey data...")
    probs = pl.DataFrame(results.predict(X.to_numpy()), schema=list(mode_dtype.categories))
    share_df = get_mode_shares(probs.to_numpy(), mode_dtype, df_survey["weight"].to_numpy())
    return results, probs, share_df


def predict_population(df_pop: pl.DataFrame, mode_dtype: pl.Enum, results, first_dep):
    print("Predicting probabilities on the synthetic population data...")
    X = get_exog_variables(df_pop, first_dep)
    probs = pl.DataFrame(results.predict(X.to_numpy()), schema=list(mode_dtype.categories))
    share_df = get_mode_shares(probs.to_numpy(), mode_dtype)
    return probs, share_df


def get_mode_shares(probs: np.ndarray, mode_dtype: pl.Enum, weights: None | np.ndarray = None):
    if weights is not None:
        probs = probs * np.atleast_2d(weights).T
    share = probs.sum(axis=0) / probs.sum()
    share_df = pl.DataFrame(
        {
            "mode": pl.Series(mode_dtype.categories, dtype=mode_dtype),
            "share": share,
        }
    )
    return share_df


def compare_frequencies(
    df_survey: pl.DataFrame, survey_predicted_share: pl.DataFrame, pop_predicted_share: pl.DataFrame
):
    survey_actual_share = (
        df_survey.group_by("mode")
        .agg(survey_actual_share=pl.col("weight").sum() / df_survey["weight"].sum())
        .sort("mode")
    )
    df = survey_actual_share.join(
        survey_predicted_share.rename({"share": "survey_predicted_share"}), on="mode"
    ).join(pop_predicted_share.rename({"share": "pop_predicted_share"}), on="mode")
    print("Actual and predicted mode share:")
    print(df)


def draw_modes(probs: pl.DataFrame, mode_dtype: pl.Enum, random_seed: None | int):
    print("Drawing modes from the probabilities...")
    rng = np.random.default_rng(random_seed)
    draws = rng.random(size=len(probs))
    cum_probs = np.cumsum(probs.to_numpy(), axis=1)
    idx = np.empty(len(draws), dtype=np.uint8)
    for i, (p, draw) in enumerate(zip(cum_probs, draws)):
        idx[i] = np.searchsorted(p, draw)
    drawn_modes = (
        pl.Series(idx)
        .replace_strict(list(range(len(mode_dtype.categories))), mode_dtype.categories)
        .cast(mode_dtype)
    )
    return drawn_modes


def save_drawn_modes(df_pop: pl.DataFrame, drawn_modes: pl.Series, directory: str):
    print("Saving file `{}`".format(os.path.join(directory, "trip_modes.parquet")))
    df = pl.DataFrame(
        {
            "trip_id": df_pop["trip_id"],
            "mode": drawn_modes,
        }
    )
    metro_io.save_dataframe(df, os.path.join(directory, "trip_modes.parquet"))
    return df


def plot_share_mode(
    ax,
    df_survey: pl.DataFrame,
    df_pop: pl.DataFrame,
    bins: np.ndarray,
    title: str,
):
    dist_groups = pl.col("od_distance").cut(list(bins[:-1]), left_closed=True)
    survey_actual_car_share = (
        df_survey.group_by(dist_groups)
        .agg(
            share=((pl.col("mode") == "car_driver") * pl.col("weight")).sum()
            / pl.col("weight").sum()
        )
        .sort("od_distance")
    )
    survey_predicted_car_share = (
        df_survey.group_by(dist_groups)
        .agg(share=(pl.col("car_driver") * pl.col("weight")).sum() / pl.col("weight").sum())
        .sort("od_distance")
    )
    pop_predicted_car_share = (
        df_pop.group_by(dist_groups).agg(share=pl.col("car_driver").mean()).sort("od_distance")
    )
    xs = (bins[:-1] + bins[1:]) / 2
    ax.plot(
        xs,
        survey_actual_car_share["share"],
        marker="o",
        linestyle="-",
        label="Survey actual share",
        color=mpl.CMP(0),
        alpha=0.7,
    )
    ax.plot(
        xs,
        survey_predicted_car_share["share"],
        marker="o",
        linestyle="-",
        label="Survey predicted share",
        color=mpl.CMP(1),
        alpha=0.7,
    )
    ax.plot(
        xs,
        pop_predicted_car_share["share"],
        marker="o",
        linestyle="-",
        label="Synthetic population predicted share",
        color=mpl.CMP(2),
        alpha=0.7,
    )
    ax.set_xlabel("OD distance (km)")
    ax.set_ylabel("Share of car driver")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_by_departments(
    df_survey: pl.DataFrame,
    survey_probs: pl.DataFrame,
    df_pop: pl.DataFrame,
    pop_probs: pl.DataFrame,
    bins: np.ndarray,
    col: str,
    title_template: str,
    filename: str,
):
    df_survey = pl.concat((df_survey, survey_probs), how="horizontal")
    survey_data = df_survey.partition_by(col, as_dict=True)
    df_pop = pl.concat((df_pop, pop_probs), how="horizontal")
    pop_data = df_pop.partition_by(col, as_dict=True)
    departements = df_survey[col].unique().sort()
    nb_dep = len(departements)
    fig, axs = plt.subplots(nrows=nb_dep, figsize=mpl.set_size(fraction=0.8, ratio=0.618 * nb_dep))
    for i, dpmt in enumerate(departements):
        subdf_survey = survey_data[(dpmt,)]
        subdf_pop = pop_data[(dpmt,)]
        plot_share_mode(
            axs[i],
            subdf_survey,
            subdf_pop,
            bins,
            title_template.format(dpmt),
        )
    fig.tight_layout()
    fig.savefig(filename)


def plot_by_departure_period(
    df_survey: pl.DataFrame,
    survey_probs: pl.DataFrame,
    df_pop: pl.DataFrame,
    pop_probs: pl.DataFrame,
    bins: np.ndarray,
    filename: str,
):
    df_survey = pl.concat((df_survey, survey_probs), how="horizontal")
    df_pop = pl.concat((df_pop, pop_probs), how="horizontal")
    fig, axs = plt.subplots(nrows=4, figsize=mpl.set_size(fraction=0.8, ratio=0.618 * 4))
    plot_share_mode(
        axs[0],
        df_survey,
        df_pop,
        bins,
        "Share of car driver (all trips)",
    )
    plot_share_mode(
        axs[1],
        df_survey.filter("morning_departure"),
        df_pop.filter("morning_departure"),
        bins,
        "Share of car driver (morning departure)",
    )
    plot_share_mode(
        axs[2],
        df_survey.filter("evening_departure"),
        df_pop.filter("evening_departure"),
        bins,
        "Share of car driver (evening departure)",
    )
    plot_share_mode(
        axs[3],
        df_survey.filter((~pl.col("morning_departure") & (~pl.col("evening_departure")))),
        df_pop.filter((~pl.col("morning_departure") & (~pl.col("evening_departure")))),
        bins,
        "Share of car driver (other departure)",
    )
    fig.tight_layout()
    fig.savefig(filename)


def plot_graphs(
    df_survey: pl.DataFrame,
    survey_probs: pl.DataFrame,
    df_pop: pl.DataFrame,
    pop_probs: pl.DataFrame,
    graph_dir: str,
):
    print("Generating graphs...")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    bins = np.quantile(df_survey["od_distance"].to_numpy(), np.linspace(0.0, 1.0, 21))
    plot_by_departments(
        df_survey,
        survey_probs,
        df_pop,
        pop_probs,
        bins,
        "departement_origin",
        "Share of car driver (Origin département {})",
        os.path.join(graph_dir, "car_driver_share_departement_origin.png"),
    )
    plot_by_departments(
        df_survey,
        survey_probs,
        df_pop,
        pop_probs,
        bins,
        "departement_destination",
        "Share of car driver (Destination département {})",
        os.path.join(graph_dir, "car_driver_share_departement_destination.png"),
    )
    plot_by_departure_period(
        df_survey,
        survey_probs,
        df_pop,
        pop_probs,
        bins,
        os.path.join(graph_dir, "car_driver_share_departure_period.png"),
    )


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "travel_survey.directory",
        "population_directory",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    mode_dtype = pl.Enum(
        ["public_transit", "car_driver", "car_passenger", "motorcycle", "bicycle", "walking"]
    )

    directory = config["travel_survey"]["directory"]
    survey_type = config["travel_survey"]["survey_type"]
    if survey_type == "EGT":
        df_survey = read_egt(directory, mode_dtype)
    else:
        raise Exception(f"Error. Unsupported survey type: {survey_type}")

    df_pop = read_population(config["population_directory"], df_survey)

    first_dep = min(
        df_survey.select(pl.col("departement_origin").min()).item(),
        df_survey.select(pl.col("departement_destination").min()).item(),
        df_pop.select(pl.col("departement_origin").min()).item(),
        df_pop.select(pl.col("departement_destination").min()).item(),
    )

    results, survey_probs, survey_predicted_share = estimate_model(df_survey, mode_dtype, first_dep)

    pop_probs, pop_predicted_share = predict_population(df_pop, mode_dtype, results, first_dep)

    compare_frequencies(df_survey, survey_predicted_share, pop_predicted_share)

    drawn_modes = draw_modes(pop_probs, mode_dtype, config.get("random_seed"))

    df = save_drawn_modes(df_pop, drawn_modes, config["population_directory"])

    if config["synthetic_population"].get("predict_modes", dict()).get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "synthetic_population.mode_prediction")
        plot_graphs(
            df_survey,
            survey_probs,
            df_pop,
            pop_probs,
            graph_dir,
        )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
