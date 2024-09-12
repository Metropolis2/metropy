import os
import time

import numpy as np
import polars as pl

import metropy.utils.io as metro_io


def read_trips(directory: str):
    print("Reading trips...")
    trips = metro_io.scan_dataframe(os.path.join(directory, "trips.parquet"))
    persons = metro_io.scan_dataframe(os.path.join(directory, "persons.parquet"))
    trips = trips.join(persons, on="person_id")
    zones = metro_io.scan_dataframe(os.path.join(directory, "trip_zones.parquet")).with_columns(
        # TODO. This is specific to IDF and should be improved.
        pl.col("departement_destination")
        .replace_strict(
            {"75": 1, "77": 3, "78": 3, "91": 3, "92": 2, "93": 2, "94": 2, "95": 3}, default=3
        )
        .alias("destination_area")
    )
    trips = trips.join(zones, on="trip_id")
    trips = trips.filter(
        pl.col("preceding_purpose") == "home",
        pl.col("following_purpose") == "work",
        pl.col("socioprofessional_class").is_in((3, 4, 5, 6)),
        pl.len().over("tour_id") == 2,
    )
    trips = trips.select("trip_id", "destination_area", "socioprofessional_class")
    df = trips.collect()
    print(f"Number of trips to be considered: {len(df):,}")
    return df


def read_probabilities(filename: str):
    return metro_io.read_dataframe(
        filename,
        columns=["destination_area", "socioprofessional_class", "start_time", "duration", "weight"],
    )


def draw_desired_times(trips: pl.DataFrame, probs: pl.DataFrame, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = pl.DataFrame()
    grouped_trips = trips.partition_by("destination_area", "socioprofessional_class", as_dict=True)
    grouped_probs = probs.partition_by("destination_area", "socioprofessional_class", as_dict=True)
    for (dest_area, spc), subtrips in grouped_trips.items():
        n = len(subtrips)
        p = grouped_probs.get((dest_area, spc))
        if p is None:
            raise Exception(
                "No desired times found for destination area {} and socio-professional class {}".format(
                    dest_area, spc
                )
            )
        idx = rng.choice(len(p), size=n, replace=True, p=p["weight"] / p["weight"].sum())
        subdf = pl.concat(
            (p.select("start_time", "duration")[idx], pl.DataFrame(subtrips["trip_id"])),
            how="horizontal",
        )
        df = pl.concat((df, subdf), how="diagonal_relaxed")
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "travel_survey.desired_times.desired_work_times_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()
    trips = read_trips(config["population_directory"])
    probs = read_probabilities(
        config["travel_survey"]["desired_times"]["desired_work_times_filename"]
    )
    df = draw_desired_times(trips, probs, config.get("random_seed"))
    # TODO. Add graph.
    metro_io.save_dataframe(
        df, os.path.join(config["population_directory"], "desired_times.parquet")
    )
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
