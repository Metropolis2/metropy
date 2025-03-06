import os
import time

import numpy as np
import polars as pl
import pandas as pd
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


def read_municipalities(insee_filename: str, arrondissement_filename: str | None):
    print("Reading municipalities...")
    gdf = metro_io.read_geodataframe(insee_filename, columns=["INSEE_COM", "geometry"])
    gdf.rename(columns={"INSEE_COM": "insee"}, inplace=True)
    gdf["insee"] = gdf["insee"].astype(str)
    gdf["priority"] = 1
    if arrondissement_filename is not None:
        print("Reading arrondissements...")
        arr_gdf = metro_io.read_geodataframe(
            arrondissement_filename, columns=["INSEE_ARM", "geometry"]
        )
        arr_gdf.rename(columns={"INSEE_ARM": "insee"}, inplace=True)
        arr_gdf["insee"] = arr_gdf["insee"].astype(str)
        arr_gdf["priority"] = 2
        gdf = gpd.GeoDataFrame(pd.concat((gdf, arr_gdf)))
    return gdf


def match_home_to_municipalities(
    households: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame, crs: str
):
    print("Matching household homes to municipalities...")
    households.to_crs(crs, inplace=True)
    municipalities.to_crs(crs, inplace=True)
    households = households.sjoin(municipalities, how="left", predicate="intersects")
    df = pl.from_pandas(
        households.loc[:, ["household_id", "number_of_vehicles", "insee", "priority"]]
    )
    # Duplicates can happen if a home is located in two municipalities, we only keep one match, with
    # highest priority.
    df = df.sort("priority").unique(subset=["household_id"], keep="last")
    if df["insee"].is_null().any():
        n = df["insee"].is_null().sum()
        print("Warning: {} households whose home is not located in any municipality".format(n))
        df = df.drop_nulls()
    print("Found {:,} municipalities with at least one household".format(df["insee"].n_unique()))
    return df


def read_vehicle_fleet(filename: str, households: pl.DataFrame, year: int):
    print("Reading vehicle fleet...")
    if filename.endswith(".csv"):
        vehicles = pl.scan_csv(
            filename,
            separator=";",
            skip_rows=1,
            schema_overrides={"COMMUNE_CODE": pl.String, f"PARC_{year}": pl.UInt64},
        )
        vehicles = vehicles.rename({"COMMUNE_CODE": "insee", f"PARC_{year}": "nb_vehicles"})
        # Consider only cars.
        vehicles = vehicles.filter(pl.col("CLASSE_VEHICULE") == "vp")
    elif filename.endswith(".xlsx"):
        vehicles = pl.read_excel(
            filename,
            read_options={"header_row": 3},
            schema_overrides={"Code commune de résidence": pl.String, f"{year}": pl.UInt64},
        ).lazy()
        vehicles = vehicles.rename(
            {
                "Code commune de résidence": "insee",
                "Carburant": "CARBURANT",
                "Crit'Air": "CRITAIR",
                f"{year}": "nb_vehicles",
            }
        )
        # Consider only household cars (i.e., exclude professional cars).
        vehicles = vehicles.filter(pl.col("statut") == "PAR")
        vehicles = vehicles.filter(pl.col("CARBURANT") != "Non déterminé")
    else:
        raise Exception(f"Invalid fleet filename: `{filename}`")
    # Select the communes of interest.
    vehicles = vehicles.filter(pl.col("insee").is_in(households["insee"]))
    # Drop "Inconnu".
    vehicles = vehicles.filter(pl.col("CRITAIR") != "Inconnu")
    # Normalize names.
    vehicles = vehicles.with_columns(
        pl.col("CARBURANT")
        .str.to_lowercase()
        .str.replace_all(" ", "_")
        .str.replace_all("'", "")
        .str.replace_all("é", "e")
        .str.replace_all("è", "e"),
        pl.col("CRITAIR")
        .str.to_lowercase()
        .str.replace_all(" ", "_")
        .str.replace_all("'", "")
        .str.replace_all("é", "e")
        .str.replace_all("è", "e"),
    )
    # Set CARBURANT and CRITAIR as categorical variables.
    vehicles = vehicles.with_columns(
        pl.col("CARBURANT").cast(pl.Categorical),
        pl.col("CRITAIR").cast(pl.Categorical),
    )
    vehicles = vehicles.with_columns(
        share=pl.col("nb_vehicles") / pl.col("nb_vehicles").sum().over("insee"),
    )
    vehicles = vehicles.select("insee", "CARBURANT", "CRITAIR", "nb_vehicles", "share")
    vehicles = vehicles.collect()
    # Create zero values for vehicle types that do not appear in an insee.
    unique_insee = vehicles.select("insee").unique()
    unique_types = vehicles.select("CARBURANT", "CRITAIR").unique()
    base_df = unique_insee.join(unique_types, how="cross")
    vehicles = base_df.join(
        vehicles, on=["insee", "CARBURANT", "CRITAIR"], how="left"
    ).with_columns(pl.col("nb_vehicles").fill_null(0), pl.col("share").fill_null(0.0))
    print("Total number of vehicles in the fleet: {:,}".format(vehicles["nb_vehicles"].sum()))
    return vehicles


def make_fleet_evolution(vehicles: pl.DataFrame, evolution_matrix_filename: str):
    evolution_matrix = metro_io.read_dataframe(evolution_matrix_filename)
    evolution_matrix = evolution_matrix.with_columns(
        pl.col("from_critair")
        .str.to_lowercase()
        .str.replace_all(" ", "_")
        .str.replace_all("'", "")
        .str.replace_all("é", "e")
        .str.replace_all("è", "e"),
        pl.col("to_critair")
        .str.to_lowercase()
        .str.replace_all(" ", "_")
        .str.replace_all("'", "")
        .str.replace_all("é", "e")
        .str.replace_all("è", "e"),
    )
    # Aggregate Crit'Air counts by INSEE.
    n0 = vehicles["nb_vehicles"].sum()
    critair_counts = (
        vehicles.group_by("insee", "CRITAIR")
        .agg(pl.col("nb_vehicles").sum())
        .with_columns(new_nb_vehicles=pl.col("nb_vehicles"))
    )
    for from_critair, to_critair, coef in zip(
        evolution_matrix["from_critair"], evolution_matrix["to_critair"], evolution_matrix["coef"]
    ):
        shifts = critair_counts.filter(pl.col("CRITAIR") == from_critair).select(
            "insee", shift=pl.col("nb_vehicles") * coef
        )
        critair_counts = (
            critair_counts.join(shifts, on="insee", how="left")
            .with_columns(
                new_nb_vehicles=pl.col("new_nb_vehicles")
                + pl.when(pl.col("CRITAIR") == from_critair).then(-pl.col("shift")).otherwise(0.0)
                + pl.when(pl.col("CRITAIR") == to_critair).then("shift").otherwise(0.0)
            )
            .drop("shift")
        )
    # The new number of vehicles for a tuple INSEE, CARBURANT, CRITAIR is the new total number of
    # vehicles for the pair INSEE, CRITAIR (computed in `critair_counts) multiplied by the observed
    # share of INSEE, CARBURANT, CRITAIR tuples within the INSEE, CRITAIR pair.
    vehicles = vehicles.with_columns(
        default_critair_share=pl.col("nb_vehicles").sum().over("CARBURANT", "CRITAIR")
        / pl.col("nb_vehicles").sum().over("CRITAIR")
    )
    vehicles = vehicles.with_columns(
        critair_share=pl.col("nb_vehicles")
        .truediv(pl.col("nb_vehicles").sum().over("insee", "CRITAIR"))
        .fill_nan(pl.col("default_critair_share"))
    )
    vehicles = vehicles.join(critair_counts.drop("nb_vehicles"), on=["insee", "CRITAIR"])
    vehicles = vehicles.with_columns(
        nb_vehicles=pl.col("new_nb_vehicles") * pl.col("critair_share")
    )
    assert n0 == round(vehicles["nb_vehicles"].sum(), 0)
    vehicles = vehicles.with_columns(
        share=pl.col("nb_vehicles") / pl.col("nb_vehicles").sum().over("insee")
    )
    print(
        vehicles.group_by("CRITAIR")
        .agg(pl.col("nb_vehicles").sum())
        .with_columns(share=pl.col("nb_vehicles") / pl.col("nb_vehicles").sum())
        .sort("CRITAIR")
    )
    return vehicles.select("insee", "CARBURANT", "CRITAIR", "nb_vehicles", "share")


def draw_vehicles(households: pl.DataFrame, vehicles: pl.DataFrame, random_seed=None):
    rng = np.random.default_rng(random_seed)
    households = households.sort("household_id")
    vehicles = vehicles.sort("insee", "CARBURANT", "CRITAIR")
    unknown_insee = set(households["insee"]).difference(set(vehicles["insee"]))
    if unknown_insee:
        print(f"Warning. Unknown fleet for {len(unknown_insee)} municipalities: {unknown_insee}")
        n0 = households["number_of_vehicles"].sum()
        households = households.filter(pl.col("insee").is_in(unknown_insee).not_())
        n = households["number_of_vehicles"].sum()
        print(f"representing {n0-n:,} vehicles ({(n0-n)/n0:.2%} of total)")
    print("Drawing vehicles for each household...")
    vehicles_by_insee = vehicles.partition_by(["insee"], as_dict=True)
    n = int(households["number_of_vehicles"].sum())
    drawn_household_ids = np.empty(n, dtype=np.uint64)
    drawn_fuel_types = np.empty(n, dtype=np.uint16)
    drawn_critairs = np.empty(n, dtype=np.uint16)
    i = 0
    for (insee_code,), households in households.partition_by(["insee"], as_dict=True).items():
        m = int(households["number_of_vehicles"].sum())
        pool = vehicles_by_insee[(insee_code,)]
        probs = pool["share"]
        draws = rng.choice(np.arange(len(pool)), size=m, p=probs, replace=True)
        drawn_household_ids[i : i + m] = np.repeat(
            households["household_id"], households["number_of_vehicles"]
        )
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
        pl.col("fuel_type").replace_strict(
            list(range(len(fuel_type_cats))),
            fuel_type_cats,
            return_dtype=fuel_type_cats.dtype,
        ),
        pl.col("critair").replace_strict(
            list(range(len(critair_cats))),
            critair_cats,
            return_dtype=critair_cats.dtype,
        ),
    )
    print(
        "Fuel shares:\n{}".format(
            df.group_by("fuel_type").agg((pl.len() / len(df)).alias("share")).sort("share")
        )
    )
    print(
        "Critair shares:\n{}".format(
            df.group_by("critair").agg((pl.len() / len(df)).alias("share")).sort("critair")
        )
    )
    return df


def add_european_standard(df: pl.DataFrame, car_types: dict):
    data = list()
    for fuel_type, ft_dict in car_types.items():
        for critair, euro_std in ft_dict.items():
            data.append((fuel_type, critair, euro_std))
    car_type_df = pl.DataFrame(
        data,
        schema={
            "fuel_type": pl.String,
            "critair": pl.String,
            "euro_standard": pl.String,
        },
        orient="row",
    )
    df = df.join(car_type_df, on=["fuel_type", "critair"], how="left")
    if df["euro_standard"].is_null().any():
        print("Warning: Euro standard is unknown for the following car types:")
        null_df = (
            df.filter(pl.col("euro_standard").is_null())
            .unique(subset=["fuel_type", "critair"])
            .select("fuel_type", "critair")
        )
        print(null_df)
    print(
        "Euro standard shares:\n{}".format(
            df.group_by("euro_standard").agg((pl.len() / len(df)).alias("share")).sort("share")
        )
    )
    return df


def save_critairs_by_insee(df: pl.DataFrame, households: pl.DataFrame, output_filename: str):
    critairs_by_insee = (
        df.join(households.select("household_id", "insee"), on="household_id")
        .group_by("insee", "critair")
        .agg(pl.len())
        .with_columns(share=pl.col("len") / pl.col("len").sum().over("insee"))
        .pivot(on="critair", index="insee", sort_columns=True)
        .fill_null(0)
        .sort("insee")
    )
    critairs_by_insee.write_parquet(output_filename)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "crs",
        "france.insee_filename",
        "synthetic_population.french_vehicle_fleet.fleet_filename",
        "synthetic_population.french_vehicle_fleet.fleet_year",
        "synthetic_population.french_vehicle_fleet.car_types",
    ]
    check_keys(config, mandatory_keys)
    this_config = config["synthetic_population"]["french_vehicle_fleet"]

    t0 = time.time()

    households = read_households(config["population_directory"])

    if config["synthetic_population"]["french_vehicle_fleet"].get("use_arrondissement", False):
        arrondissement_filename = config["france"]["arrondissement_filename"]
    else:
        arrondissement_filename = None
    municipalities = read_municipalities(
        config["france"]["insee_filename"], arrondissement_filename
    )

    households = match_home_to_municipalities(households, municipalities, config["crs"])

    vehicles = read_vehicle_fleet(
        this_config["fleet_filename"], households, this_config["fleet_year"]
    )

    if "evolution_matrix" in config["synthetic_population"]["french_vehicle_fleet"]:
        vehicles = make_fleet_evolution(
            vehicles, config["synthetic_population"]["french_vehicle_fleet"]["evolution_matrix"]
        )

    df = draw_vehicles(households, vehicles, config.get("random_seed"))

    df = add_european_standard(df, this_config["car_types"])

    metro_io.save_dataframe(df, os.path.join(config["population_directory"], "vehicles.parquet"))

    save_critairs_by_insee(
        df, households, os.path.join(config["population_directory"], "vehicles_by_insee.parquet")
    )

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
