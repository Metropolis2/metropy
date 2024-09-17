import os
import time

import polars as pl


def read_egt(directory: str, period: list[float]):
    print("Reading EGT...")
    lf = pl.scan_csv(os.path.join(directory, "Format_csv", "Deplacements_semaine.csv"))
    # Rename columns.
    lf = lf.rename(
        {
            "DPORTEE": "od_distance",
            "POIDSP": "weight",
            "ORCOUR": "origin_area",
            "DESTCOUR": "destination_area",
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
    # Create mode column.
    MODE_MAP = {
        1: "public_transit",
        2: "car_driver",
        3: "car_passenger",
        4: "motorcycle",
        5: "bicycle",
        6: "other",
        7: "walking",
    }
    lf = lf.with_columns(pl.col("MODP_H7").replace_strict(MODE_MAP).alias("mode"))
    # Filter out invalid modes.
    lf = lf.filter(pl.col("mode") != "other")
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
    # Filter out invalid tours.
    lf = lf.filter(
        pl.col("preceding_purpose").first().over("person_id") == "home",
        pl.col("following_purpose").last().over("person_id") == "home",
    )
    # Create tour_id.
    lf = lf.with_columns(
        pl.when(pl.col("preceding_purpose") == "home")
        .then(pl.lit(1))
        .otherwise(None)
        .cast(pl.UInt64)
        .cum_count()
        .alias("tour_id")
    )
    # Join with person data.
    lf_persons = pl.scan_csv(os.path.join(directory, "Format_csv", "Personnes_semaine.csv"))
    lf_persons = lf_persons.with_columns((pl.col("NQUEST") * 10 + pl.col("NP")).alias("person_id"))
    lf_persons = lf_persons.rename({"CS8": "socioprofessional_class"})
    lf = lf.join(lf_persons, on="person_id")
    lf = lf.select(
        "person_id",
        "tour_id",
        "trip_id",
        "departure_time",
        "arrival_time",
        "mode",
        "preceding_purpose",
        "following_purpose",
        "od_distance",
        "socioprofessional_class",
        "origin_area",
        "destination_area",
        "weight",
    )
    df = lf.collect()
    print("Number of trips: {:,}".format(len(df)))
    return df


def get_desired_work_times(df: pl.DataFrame):
    # Compute activity time.
    df = df.with_columns(
        (pl.col("departure_time").shift(-1).over("tour_id") - pl.col("arrival_time")).alias(
            "activity_time"
        )
    ).with_columns(pl.col("activity_time").sum().over("tour_id").alias("total_activity_time"))
    # Filter out tours with negative activity times.
    df = df.filter(pl.col("activity_time").min().over("tour_id") >= 0)
    # Compute probability of joint start time and activity time for work activities (when work is
    # the only activity of the tour).
    # We select only walking and bicycle trips, where we know that start time could be chosen (no
    # congestion).
    # We select only socio-professional classes 3, 4, 5, 6, where there is enough observations.
    desired_work_times = (
        df.lazy()
        .filter(
            pl.col("preceding_purpose") == "home",
            pl.col("following_purpose") == "work",
            pl.col("mode").is_in(("walking", "bicycle")),
            pl.col("socioprofessional_class").is_in((3, 4, 5, 6)),
            pl.len().over("tour_id") == 2,
        )
        .rename({"arrival_time": "start_time", "activity_time": "duration"})
        .select("destination_area", "socioprofessional_class", "start_time", "duration", "weight")
        .collect()
    )
    return desired_work_times


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    import metropy.utils.io as metro_io

    config = read_config()
    mandatory_keys = [
        "travel_survey.directory",
        "travel_survey.survey_type",
        "travel_survey.desired_times.desired_work_times_filename",
        "run.period",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()
    directory = config["travel_survey"]["directory"]
    survey_type = config["travel_survey"]["survey_type"]
    if survey_type == "EGT":
        df = read_egt(directory, config["run"]["period"])
    else:
        raise Exception(f"Error. Unsupported survey type: {survey_type}")

    desired_work_times = get_desired_work_times(df)

    metro_io.save_dataframe(
        desired_work_times, config["travel_survey"]["desired_times"]["desired_work_times_filename"]
    )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
