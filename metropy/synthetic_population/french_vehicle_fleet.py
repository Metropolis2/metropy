import os
import time

import numpy as np
import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io


def read_households(population_directory: str):
    print("Reading households...")
    households = metro_io.read_geodataframe(
        os.path.join(population_directory, "households.parquet"),
        columns=["household_id", "number_of_vehicles", "geometry"],
    )
    households = households.loc[households["number_of_vehicles"] > 0]
    print("Number of households with at least 1 car: {:,}".format(len(households)))
    print("Total number of cars: {:,}".format(households["number_of_vehicles"].sum()))
    return households


def read_municipalities(filename: str):
    print("Reading municipalities...")
    gdf = metro_io.read_geodataframe(filename, columns=["INSEE_COM", "geometry"])
    gdf.rename(columns={"INSEE_COM": "insee"}, inplace=True)
    gdf["insee"] = gdf["insee"].astype(str)
    return gdf


def match_home_to_municipalities(
    households: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame, crs: str
):
    print("Matching household homes to municipalities...")
    households.to_crs(crs, inplace=True)
    municipalities.to_crs(crs, inplace=True)
    households = households.sjoin(municipalities, how="left", predicate="intersects")
    df = pl.from_pandas(households.loc[:, ["household_id", "number_of_vehicles", "insee"]])
    # Duplicates can happen if a home is located in two municipalities, we only keep one match.
    df = df.unique(subset=["household_id", "insee"])
    if df["insee"].is_null().any():
        n = df["insee"].is_null().sum()
        print("Warning: {} households whose home is not located in any municipality".format(n))
        df = df.drop_nulls()
    print("Found {:,} municipalities with at least one household".format(df["insee"].n_unique()))
    return df


def read_vehicle_fleet(filename: str, df: pl.DataFrame, year: int):
    print("Reading vehicle fleet...")
    vehicles = pl.scan_csv(
        filename,
        separator=";",
        skip_rows=1,
        dtypes={"COMMUNE_CODE": pl.String, f"PARC_{year}": pl.UInt64},
    )
    vehicles = vehicles.rename({"COMMUNE_CODE": "insee", f"PARC_{year}": "nb_vehicles"})
    # Select the communes of interest.
    vehicles = vehicles.filter(pl.col("insee").is_in(df["insee"]))
    # Drop "Inconnu".
    vehicles = vehicles.filter(pl.col("CRITAIR") != "Inconnu")
    # Consider only cars.
    vehicles = vehicles.filter(pl.col("CLASSE_VEHICULE") == "vp")
    # Set CARBURANT and CRITAIR as categorical variables.
    vehicles = vehicles.with_columns(
        pl.col("CARBURANT").cast(pl.Categorical),
        pl.col("CRITAIR").cast(pl.Categorical),
    )
    vehicles = vehicles.select("insee", "CARBURANT", "CRITAIR", "nb_vehicles")
    vehicles = vehicles.collect()
    print("Total number of vehicles in the fleet: {:,}".format(vehicles["nb_vehicles"].sum()))
    return vehicles


def draw_vehicles(df: pl.DataFrame, vehicles: pl.DataFrame):
    print("Drawing vehicles for each household...")
    vehicles_by_insee = vehicles.partition_by(["insee"], as_dict=True)
    n = int(df["number_of_vehicles"].sum())
    drawn_household_ids = np.empty(n, dtype=np.uint64)
    drawn_fuel_types = np.empty(n, dtype=np.uint16)
    drawn_critairs = np.empty(n, dtype=np.uint16)
    i = 0
    for (insee_code,), df in df.partition_by(["insee"], as_dict=True).items():
        m = int(df["number_of_vehicles"].sum())
        pool = vehicles_by_insee[(insee_code,)]
        probs = pool["nb_vehicles"] / pool["nb_vehicles"].sum()
        draws = np.random.choice(np.arange(len(pool)), size=m, p=probs, replace=True)
        drawn_household_ids[i : i + m] = np.repeat(df["household_id"], df["number_of_vehicles"])
        drawn_fuel_types[i : i + m] = pool["CARBURANT"].to_physical()[draws]
        drawn_critairs[i : i + m] = pool["CRITAIR"].to_physical()[draws]
        i += m
    df = pl.DataFrame(
        data={
            "household_id": drawn_household_ids,
            "fuel_type": drawn_fuel_types,
            "critair": drawn_critairs,
        }
    )
    fuel_type_cats = vehicles["CARBURANT"].cat.get_categories()
    critair_cats = vehicles["CRITAIR"].cat.get_categories()
    df = df.with_columns(
        pl.col("fuel_type")
        .replace(list(range(len(fuel_type_cats))), fuel_type_cats)
        .cast(pl.Categorical),
        pl.col("critair")
        .replace(list(range(len(critair_cats))), critair_cats)
        .cast(pl.Categorical),
    )
    print(
        "Fuel shares:\n{}".format(
            df.group_by("fuel_type").agg((pl.len() / len(df)).alias("share")).sort("share")
        )
    )
    print(
        "Critair shares:\n{}".format(
            df.group_by("critair").agg((pl.len() / len(df)).alias("share")).sort("share")
        )
    )
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "crs",
        "synthetic_population.french_vehicle_fleet.insee_filename",
        "synthetic_population.french_vehicle_fleet.fleet_filename",
        "synthetic_population.french_vehicle_fleet.fleet_year",
        "synthetic_population.french_vehicle_fleet.output_filename",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["synthetic_population"]["french_vehicle_fleet"]

    t0 = time.time()

    households = read_households(config["population_directory"])

    municipalities = read_municipalities(this_config["insee_filename"])

    df = match_home_to_municipalities(households, municipalities, config["crs"])

    vehicles = read_vehicle_fleet(this_config["fleet_filename"], df, this_config["fleet_year"])

    df = draw_vehicles(df, vehicles)

    metro_io.save_dataframe(
        df,
        this_config["output_filename"],
    )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
