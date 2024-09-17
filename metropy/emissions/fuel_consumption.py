import os
import time

import polars as pl

import metropy.utils.io as metro_io


def read_routes(filename: str, population_dir: str):
    print("Reading routes...")
    #  df = metro_io.read_dataframe(filename, columns=["trip_id", "route"])
    df = pl.read_parquet(filename, columns=["trip_id", "route"], n_rows=10000)
    trips = metro_io.read_dataframe(
        os.path.join(population_dir, "trips.parquet"), columns=["person_id", "trip_id"]
    )
    df = df.join(trips, on="trip_id")
    return df


def read_edges(
    filename: str,
    edge_penalties_filename: str | None,
):
    print("Reading edges...")
    df = metro_io.read_dataframe(
        filename,
        columns=["edge_id", "length", "speed_limit"],
    ).rename({"length": "length_m"})
    if edge_penalties_filename is not None:
        penalties = metro_io.read_dataframe(edge_penalties_filename)
        if "speed" in penalties.columns:
            print("Using multiplicative penalties")
            df = df.join(
                penalties.select("edge_id", "speed"), on="edge_id", how="left"
            ).with_columns(pl.col("speed").fill_null(pl.col("speed_limit")))
        else:
            df = df.with_columns(speed=pl.col("speed_limit"))
        if "additive_penalty" in penalties.columns:
            print("Using additive penalties")
            df = df.join(
                penalties.select("edge_id", "additive_penalty"), on="edge_id", how="left"
            ).with_columns(pl.col("additive_penalty").fill_null(0.0))
        else:
            df = df.with_columns(additive_penalty=pl.lit(0.0))
    df = (
        df.with_columns(
            travel_time_sec=pl.col("length_m") / (pl.col("speed") / 3.6)
            + pl.col("additive_penalty"),
            length_km=pl.col("length_m") / 1000.0,
        )
        .with_columns(speed_kph=pl.col("length_km") / (pl.col("travel_time_sec") / 3600))
        .select("edge_id", "length_km", "speed_kph")
    )
    return df


def read_vehicles(population_dir: str):
    print("Reading vehicles...")
    lf_persons = pl.scan_parquet(os.path.join(population_dir, "persons.parquet"))
    lf_vehicles = pl.scan_parquet(os.path.join(population_dir, "vehicles.parquet"))
    # TODO: For now, only the first vehicle is used.
    lf_vehicles = lf_vehicles.unique(subset=["household_id"], keep="first")
    lf = lf_persons.join(lf_vehicles, on="household_id")
    lf = lf.select("household_id", "person_id", "fuel_type", "critair", "euro_standard")
    return lf.collect()


def merge_with_edges(df_routes: pl.DataFrame, df_edges: pl.DataFrame):
    print("Merging routes with edges...")
    lf = (
        df_routes.lazy()
        .rename({"route": "edge_id"})
        .explode("edge_id")
        .join(df_edges.lazy(), on="edge_id")
    )
    return lf


def merge_with_vehicles(lf: pl.LazyFrame, df_vehicles: pl.DataFrame):
    print("Merging persons with vehicles...")
    # At this point, the persons with no vehicle are dropped.
    lf = lf.join(df_vehicles.lazy(), on="person_id", how="inner")
    return lf


def clean(lf: pl.LazyFrame, config: dict):
    # Kilometers traveled since departure.
    lf = lf.with_columns(
        pl.col("length_km").cum_sum().shift(1).fill_null(0.0).over("trip_id").alias("km_since_dep")
    )
    # Find speed categories.
    lf = lf.with_columns(
        (5 * ((pl.col("speed_kph") - 0.5) // 5) + 5).cast(pl.UInt8).alias("speed_cat")
    )
    # Flag hot emissions.
    lf = lf.with_columns(
        (pl.col("km_since_dep") >= config.get("cold_emissions_threshold", 0.0)).alias("hot")
    )
    return lf


def read_emission_factors(config: dict):
    print("Reading emission factors...")
    lf = metro_io.scan_dataframe(config["emission_factor_file"])
    lf = lf.filter(pl.col("pollutant") == "EC")
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


def compute_emissions(lf: pl.LazyFrame, df_emissions: pl.DataFrame):
    print("Compute emissions...")
    # Add emission factors to the main LazyFrame.
    lf = lf.join(
        df_emissions.lazy(),
        on=["euro_standard", "hot", "speed_cat"],
        how="left",
    )
    # Compute emissions.
    lf = lf.with_columns(energy_consumption=pl.col("length_km") * pl.col("EC"))
    # Compute fuel consumption
    lf = lf.with_columns((pl.col("energy_consumption") * (1 / 43.2345)).alias("fuel_consumption"))
    return lf


def group_by(lf: pl.LazyFrame):
    results = lf.group_by("trip_id").agg(pl.col("fuel_consumption").sum())
    return results.collect(streaming=True)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "clean_edges_file",
        "routing.road_split.trips_filename",
        "emisens.emission_factor_file",
        "emisens.cold_emissions_threshold",
        "fuel_consumption.output_filename",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["emisens"]

    t0 = time.time()

    df_routes = read_routes(
        config["routing"]["road_split"]["trips_filename"], config["population_directory"]
    )

    df_edges = read_edges(
        config["clean_edges_file"],
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
    )

    lf = merge_with_edges(df_routes, df_edges)

    df_vehicles = read_vehicles(config["population_directory"])

    lf = merge_with_vehicles(lf, df_vehicles)

    lf = clean(lf, this_config)

    df_emissions = read_emission_factors(this_config)

    lf = compute_emissions(lf, df_emissions)

    results = group_by(lf)

    metro_io.save_dataframe(results, config["fuel_consumption"]["output_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
