import os
import time

import numpy as np
import polars as pl
import gower
import kmedoids

import metropy.utils.io as metro_io
import metropy.utils.mpl as mpl


def read_egt(directory: str, period: list[float]):
    print("Reading EGT...")
    lf = pl.scan_csv(
        os.path.join(directory, "Format_csv", "Deplacements_semaine.csv"),
        schema_overrides={"ORDEP": pl.String, "DESTDEP": pl.String},
    )
    # Rename columns.
    lf = lf.rename(
        {
            "DPORTEE": "od_distance",
            "POIDSP": "weight",
            "ORCOUR": "area_origin",
            "DESTCOUR": "area_destination",
            "ORDEP": "departement_origin",
            "DESTDEP": "departement_destination",
        }
    )
    # Create person_id and trip_id.
    lf = lf.with_columns(
        (pl.col("NQUEST") * 10 + pl.col("NP")).alias("person_id"),
        (pl.col("NQUEST") * 1000 + pl.col("NP") * 100 + pl.col("ND")).alias("trip_id"),
    )
    # Create departure time and arrival time columns.
    lf = lf.with_columns(
        (pl.col("ORH") * 3600 + pl.col("ORM") * 60).alias("departure_time"),
        (pl.col("DESTH") * 3600 + pl.col("DESTM") * 60).alias("arrival_time"),
    )
    # Select trips within the time period.
    lf = lf.filter(pl.col("departure_time") >= period[0], pl.col("departure_time") <= period[1])
    # Select trips of car drivers.
    lf = lf.filter(pl.col("MODP_H7") == 2)
    # Remove trips with origin = destination.
    lf = lf.filter(pl.col("ORC") != pl.col("DESTC"))
    # Create trip purposes.
    PURPOSE_MAP = {
        1: "home",
        2: "work",
        3: "work",
        4: "education",
        5: "shop",
        6: "other",
        7: "other",
        8: "leisure",
        9: "other",
    }
    lf = lf.with_columns(pl.col("ORMOT_H9").replace_strict(PURPOSE_MAP).alias("preceding_purpose"))
    lf = lf.with_columns(
        pl.col("DESTMOT_H9").replace_strict(PURPOSE_MAP).alias("following_purpose")
    )
    # Filter out invalid routes.
    lf = lf.filter(
        pl.col("preceding_purpose").first().over("person_id") == "home",
        pl.col("following_purpose").last().over("person_id") == "home",
    )
    lf = lf.select(
        "person_id",
        "trip_id",
        "departure_time",
        "arrival_time",
        "preceding_purpose",
        "following_purpose",
        "od_distance",
        "area_origin",
        "area_destination",
        "departement_origin",
        "departement_destination",
        "weight",
    ).drop_nulls()
    df = lf.collect()
    print("Number of trips: {:,}".format(len(df)))
    return df


def classify_trips(df: pl.DataFrame):
    print("Computing Gower distance between trips...")
    variables = [
        "preceding_purpose",
        "following_purpose",
        "od_distance",
        "area_origin",
        "area_destination",
        "departement_origin",
        "departement_destination",
    ]
    x = df.select(variables).to_pandas()
    dists = gower.gower_matrix(
        x,
        cat_features=[True, True, False, True, True, True, True],
    )
    print("Classifying trips in categories...")
    cluster_size = 1200
    nb_clusters = len(x) // cluster_size
    clustering = kmedoids.KMedoids(nb_clusters, method="fasterpam").fit(dists)
    centers = pl.from_pandas(x.loc[clustering.medoid_indices_]).with_columns(
        category=pl.int_range(nb_clusters, eager=True)
    )
    df = df.select(
        pl.Series(clustering.labels_).alias("category"),
        "departure_time",
        "weight",
    )
    print("Number of trips classified: {:,}".format(len(df)))
    return df, centers


def make_graphs(df: pl.DataFrame, period: list[float], graph_dir: str):
    tds_by_category = df.partition_by("category", include_key=False, as_dict=True)
    # Cumulative density.
    fig, ax = mpl.get_figure(fraction=0.8)
    width = 0.25
    bins = np.arange(period[0] / 3600 - width / 2, period[1] / 3600 + width / 2 + 0.1, width)
    ax.hist(
        [df["departure_time"] / 3600 for df in tds_by_category.values()],
        bins=list(bins),
        density=True,
        weights=[df["weight"] for df in tds_by_category.values()],
        cumulative=True,
        histtype="step",
        alpha=0.7,
        label=[str(key[0]) for key in tds_by_category.keys()],
    )
    ax.set_xlim(period[0] / 3600, period[1] / 3600)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative density")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "cumulative_hist_by_category.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "travel_survey.directory",
        "travel_survey.survey_type",
        "travel_survey.departure_time_distribution.distribution_filename",
        "travel_survey.departure_time_distribution.cluster_filename",
        "run.period",
        "graph_directory",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()
    directory = config["travel_survey"]["directory"]
    survey_type = config["travel_survey"]["survey_type"]
    if survey_type == "EGT":
        df = read_egt(directory, config["run"]["period"])
    else:
        raise Exception(f"Error. Unsupported survey type: {survey_type}")

    df, centers = classify_trips(df)

    metro_io.save_dataframe(
        df, config["travel_survey"]["departure_time_distribution"]["distribution_filename"]
    )
    metro_io.save_dataframe(
        centers, config["travel_survey"]["departure_time_distribution"]["cluster_filename"]
    )

    graph_dir = os.path.join(config["graph_directory"], "travel_survey.departure_time_distribution")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    make_graphs(df, config["run"]["period"], graph_dir)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
