import os
import time

import polars as pl

import metropy.utils.io as metro_io

####################
###### SETUP #######
####################

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

########################
###### FUNCTIONS #######
########################


def scan_route_dataframe(filename: str):
    print("Reading routes...")
    lf = metro_io.scan_dataframe(filename)
    return lf


def read_edges(filename: str):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(filename, columns=["edge_id", "length"])
    # Set length in km.
    gdf["length"] /= 1000
    df = pl.from_pandas(gdf, schema_overrides={"id": pl.UInt64})
    return df


def scan_vehicles(population_dir: str):
    print("Reading vehicles...")
    lf_persons = pl.scan_parquet(os.path.join(population_dir, "persons.parquet"))
    lf_vehicles = pl.scan_parquet(os.path.join(population_dir, "vehicles.parquet"))
    lf_vehicles = lf_vehicles.unique(subset=["household_id"], keep="first")
    lf = lf_persons.join(lf_vehicles, on="household_id", how="left")
    return lf


def merge_with_edges(lf_routes: pl.LazyFrame, lf_edges: pl.DataFrame):
    lf_routes = lf_routes.join(lf_edges.lazy(), on="edge_id")
    return lf_routes


def merge_with_vehicles(lf_routes: pl.LazyFrame, lf_vehicles: pl.LazyFrame):
    lf_routes = lf_routes.join(lf_vehicles, left_on="agent_id", right_on="person_id", how="left")
    #  assert not df["car_type"].is_null().any() TODO
    return lf_routes


def clean(lf: pl.LazyFrame, config: dict):
    # Compute travel time.
    lf = lf.with_columns((pl.col("exit_time") - pl.col("entry_time")).alias("tt"))
    # Time is divided in periods of one hour. The firt period starts at the start of the recording
    # period of the simulation and ends when the first hour is reached. For example, if the
    # simulation starts at 05:45, the first period runs from 05:45 to 06:00, the second period runs
    # from 06:00 to 07:00 and so on.
    lf = lf.with_columns((pl.col("exit_time") // 3600).cast(pl.UInt8).alias("time_period"))
    # Compute speed (in km/h).
    lf = lf.with_columns((pl.col("length") / (pl.col("tt") / 3600)).alias("speed"))
    # Kilometers traveled since departure.
    lf = lf.with_columns(
        pl.col("length")
        .cum_sum()
        .shift(1)
        .fill_null(0.0)
        .over("agent_id", "trip_id")
        .alias("km_since_dep")
    )
    # Find speed categories.
    lf = lf.with_columns((5 * ((pl.col("speed") - 0.5) // 5) + 5).cast(pl.UInt8).alias("speed_cat"))
    # Flag hot emissions.
    lf = lf.with_columns(
        (pl.col("km_since_dep") >= config.get("cold_emissions_threshold", 0.0)).alias("hot")
    )
    return lf


def scan_emission_factors(config: dict):
    print("Reading emission factors...")
    lf = metro_io.scan_dataframe(config["emission_factor_file"])
    lf = lf.filter(pl.col("pollutant").is_in(config["pollutants"]))
    # Unpivot the LazyFrame in a LazyFrame with columns ["car_type", "hot", "speed_cat"] and one
    # column for each pollutant with the corresponding emission factor.
    lf = lf.melt(id_vars=["car_type", "pollutant"], value_name="factor")
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
            .cast(int)
        )
        .with_columns(
            # Cold emissions are cold factor + hot factor.
            pl.when(pl.col("hot"))
            .then(pl.col("factor"))
            .otherwise(pl.col("factor").sum().over(["car_type", "pollutant", "speed_cat"]))
        )
    )
    df = (
        lf.select("car_type", "pollutant", "hot", "speed_cat", "factor")
        .collect()
        .pivot(on="pollutant", index=["car_type", "hot", "speed_cat"], values="factor")
    )
    return df


def compute_emissions(lf: pl.LazyFrame, df_emissions: pl.DataFrame, config: dict):
    print("Compute emissions...")
    lf_emissions = df_emissions.lazy()
    # Add emission factors to the main LazyFrame.
    lf = lf.join(
        lf_emissions,
        on=["car_type", "hot", "speed_cat"],
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


def group_by(lf: pl.LazyFrame, config: dict):
    agg = [pl.col(p).sum() for p in config["pollutants"]]
    if "FC" in lf.columns and "CO2" in lf.columns:
        agg.extend([pl.col("FC").sum(), pl.col("CO2").sum()])
    results = lf.group_by(["agent_id", "edge_id", "time_period"]).agg(agg).collect(streaming=True)
    return results


#################
#### SCRIPT #####
#################

if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "clean_edges_file",
        "emisens.emission_factor_file",
        "emisens.pollutants",
        "emisens.cold_emissions_threshold",
        "emisens.output_filename",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["emisens"]

    t0 = time.time()

    if "road_split" in config:
        route_filename = config["road_split"]["output_directory"]
    else:
        route_filename = config["metropolis"]["output_directory"]

    lf_routes = scan_route_dataframe(route_filename)

    lf_edges = read_edges(config["clean_edges_file"])

    lf = merge_with_edges(lf_routes, lf_edges)

    lf_vehicles = scan_vehicles(config["population_directory"])

    lf = merge_with_vehicles(lf, lf_vehicles)

    lf = clean(lf, this_config)

    lf_emissions = scan_emission_factors(this_config)

    lf = compute_emissions(lf, lf_emissions, this_config)

    results = group_by(lf, this_config)

    metro_io.save_dataframe(results, this_config["output_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
