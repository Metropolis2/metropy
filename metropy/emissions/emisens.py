import os
import time

import polars as pl

import metropy.utils.io as metro_io
import metropy.emissions.base as metro_emissions

# Parameter for the non-exhaust emissions
BW_EMISSION_FACTORS = {
    "Passenger Cars": 0.0075,
    "Light Commercial Vehicles": 0.0117,
    "L-Category": 0.0037,
}
BW_MASS_FRACTION = {
    "TSP": 1,
    "PM10": 0.98,
    "PM2_5": 0.39,
    "PM1": 0.1,
    "PM0_1": 0.080,
}  # Table-3-4
RSW_EMISSION_FACTORS = {
    "Passenger Cars": 0.0150,
    "Light Commercial Vehicles": 0.0150,
    "L-Category": 0.0060,
}
RSW_MASS_FRACTION = {"TSP": 1, "PM10": 0.5, "PM2_5": 0.27, "PM1": 0, "PM0_1": 0}  # Table-3-4
TW_EMISSION_FACTORS = {
    "Passenger Cars": 0.0107,
    "Light Commercial Vehicles": 0.0169,
    "L-Category": 0.0046,
}
TW_MASS_FRACTION = {
    "TSP": 1,
    "PM10": 0.6,
    "PM2_5": 0.42,
    "PM1": 0.06,
    "PM0_1": 0.048,
}  # Table-3-4


def read_vehicles(population_dir: str):
    print("Reading vehicles...")
    lf_persons = pl.scan_parquet(os.path.join(population_dir, "persons.parquet"))
    lf_vehicles = pl.scan_parquet(os.path.join(population_dir, "vehicles.parquet"))
    lf_vehicles = lf_vehicles.unique(subset=["household_id"], keep="first")
    lf = lf_persons.join(lf_vehicles, on="household_id", how="left")
    lf_trips = pl.scan_parquet(os.path.join(population_dir, "trips.parquet")).unique(
        subset=["person_id", "tour_id"]
    )
    lf = lf.join(lf_trips, on="person_id")
    df = lf.select("tour_id", "euro_standard").collect()
    return df


def merge_with_vehicles(routes: pl.DataFrame, vehicles: pl.DataFrame):
    routes = routes.join(vehicles, on="tour_id", how="left")
    assert not routes["euro_standard"].has_nulls()
    return routes


def clean(df: pl.DataFrame, config: dict):
    # Compute speed (in km/h).
    df = df.with_columns(speed=pl.col("length") / (pl.col("travel_time") / 3600))
    # Kilometers traveled since departure.
    df = df.with_columns(
        km_since_dep=pl.col("length").cum_sum().shift(1).fill_null(0.0).over("tour_id", "trip_id")
    )
    # Find speed categories.
    df = df.with_columns(speed_cat=(5 * ((pl.col("speed") - 0.5) // 5) + 5).cast(pl.UInt8))
    # Flag hot emissions.
    df = df.with_columns(hot=pl.col("km_since_dep") >= config.get("cold_emissions_threshold", 0.0))
    # Drop unused columns.
    df = df.drop("travel_time", "entry_time", "exit_time")
    return df


def read_emission_factors(config: dict):
    print("Reading emission factors...")
    lf = metro_io.scan_dataframe(config["emission_factor_file"])
    lf = lf.filter(pl.col("pollutant").is_in(config["pollutants"]))
    # Unpivot the LazyFrame in a LazyFrame with columns ["euro_standard", "hot", "speed_cat"] and
    # one column for each pollutant with the corresponding emission factor.
    lf = lf.unpivot(index=["euro_standard", "pollutant"], value_name="factor")
    lf = (
        lf.with_columns(pl.col("variable").str.split("_"))
        .with_columns(
            pl.col("variable").list.get(0).alias("hot"),
            pl.col("variable").list.get(1).alias("speed_cat"),
        )
        .with_columns(
            # Variable "hot" is boolean: `True` for hot emissions.
            pl.col("hot")
            == "hot"
        )
        .with_columns(
            # Variable "speed_cat" is integer: upper bound `n` of the speed category `(n-5, n]`.
            pl.col("speed_cat")
            .str.extract("[0-9]+-([0-9]+)")
            .cast(pl.UInt8)
        )
        .with_columns(
            # Cold emissions are cold factor + hot factor.
            pl.when(pl.col("hot"))
            .then(pl.col("factor"))
            .otherwise(pl.col("factor").sum().over(["euro_standard", "pollutant", "speed_cat"]))
        )
    )
    df = (
        lf.select("euro_standard", "pollutant", "hot", "speed_cat", "factor")
        .collect()
        .pivot(on="pollutant", index=["euro_standard", "hot", "speed_cat"], values="factor")
    )
    return df


def compute_emissions(df: pl.DataFrame, df_emissions: pl.DataFrame, config: dict):
    print("Compute emissions...")
    # Add emission factors to the main LazyFrame.
    lf = df.lazy().join(
        df_emissions.lazy(),
        on=["euro_standard", "hot", "speed_cat"],
        how="left",
    )
    # Compute emissions.
    lf = lf.with_columns((pl.col("length") * pl.col(p)).alias(p) for p in config["pollutants"])
    if "PM" in config["pollutants"]:
        # Compute non-exhaust emissions
        lf = lf.with_columns(
            pl.when(pl.col("speed") < 40)
            .then(pl.lit(1.39))
            .otherwise(
                pl.when(pl.col("speed") > 90)
                .then(pl.lit(0.902))
                .otherwise(pl.lit(1.78) * pl.lit(0.00974) * pl.col("speed"))
            )
            .alias("tw_speed")
        )
        lf = lf.with_columns(
            pl.when(pl.col("speed") < 40)
            .then(pl.lit(1.67))
            .otherwise(
                pl.when(pl.col("speed") > 95)
                .then(pl.lit(0.185))
                .otherwise(pl.lit(2.75) * pl.lit(0.0270) * pl.col("speed"))
            )
            .alias("bw_speed")
        )
        lf = lf.with_columns(
            (
                (
                    TW_EMISSION_FACTORS["Passenger Cars"]
                    * TW_MASS_FRACTION["PM2_5"]
                    * pl.col("tw_speed")
                    + BW_EMISSION_FACTORS["Passenger Cars"]
                    * BW_MASS_FRACTION["PM2_5"]
                    * pl.col("bw_speed")
                    + RSW_EMISSION_FACTORS["Passenger Cars"] * RSW_MASS_FRACTION["PM2_5"]
                )
                * pl.col("length")
            ).alias("PM_ne")
        )
        lf = lf.with_columns((pl.col("PM") + pl.col("PM_ne")).alias("PM"))
    if "EC" in config["pollutants"]:
        # Compute CO2 emissions and fuel consumption
        lf = lf.with_columns((pl.col("EC") * (1 / 43.2345)).alias("FC")).with_columns(
            (pl.col("FC") * (3165 + 9.325)).alias("CO2")
        )
    return lf


def group_by(lf: pl.LazyFrame, config: dict, simulation_ratio: float):
    columns = lf.collect_schema().names()
    pollutants = config["pollutants"]
    if "FC" in columns and "CO2" in columns:
        pollutants.extend(["FC", "CO2"])
    agg = [pl.col(p).sum() for p in config["pollutants"]]
    results = (
        lf.group_by("tour_id", "trip_id", "edge_id", "time_period").agg(agg).collect(streaming=True)
    )
    if "NOx" in pollutants:
        v = results["NOx"].sum() / 1e6 / simulation_ratio
        print(f"Total NOx emissions: {v:,.3f}t")
    if "CO" in pollutants:
        v = results["CO"].sum() / 1e6 / simulation_ratio
        print(f"Total CO emissions: {v:,.3f}t")
    if "PM" in pollutants:
        v = results["PM"].sum() / 1e6 / simulation_ratio
        print(f"Total PM emissions: {v:,.3f}t")
    if "EC" in pollutants:
        v = results["EC"].sum() / 1e3 / simulation_ratio
        print(f"Total energy consumption: {v:,.0f}GJ")
    if "FC" in pollutants:
        v = results["FC"].sum() / simulation_ratio
        print(f"Total fuel consumption: {v:,.0f}L")
    if "CO2" in pollutants:
        v = results["CO2"].sum() / 1e6 / simulation_ratio
        print(f"Total CO2 emissions: {v:,.3f}t")
    trip_emissions = results.group_by("tour_id", "trip_id", "time_period").agg(agg)
    edge_emissions = results.group_by("edge_id", "time_period").agg(agg)
    return trip_emissions, edge_emissions


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "clean_edges_file",
        "run_directory",
        "routing.road_split.trips_filename",
        "emisens.emission_factor_file",
        "emisens.pollutants",
        "emisens.cold_emissions_threshold",
        "emisens.interval",
        "emisens.trip_emissions",
        "emisens.edge_emissions",
        "demand.modes",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["emisens"]

    t0 = time.time()

    edges = metro_emissions.read_edges(
        config["clean_edges_file"],
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
    )

    routes = metro_emissions.read_routes(
        config["run_directory"],
        config.get("routing", dict).get("road_split", dict).get("trips_filename"),
        edges,
        config["demand"]["modes"],
        config["emisens"]["interval"],
    )

    vehicles = read_vehicles(config["population_directory"])

    df = merge_with_vehicles(routes, vehicles)

    df = clean(df, this_config)

    df_emissions = read_emission_factors(this_config)

    lf = compute_emissions(df, df_emissions, this_config)

    trip_emissions, edge_emissions = group_by(
        lf, this_config, config.get("run", dict).get("simulation_ratio", 1.0)
    )

    metro_io.save_dataframe(trip_emissions, this_config["trip_emissions"])
    metro_io.save_dataframe(edge_emissions, this_config["edge_emissions"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
