import os
import time

from osgeo import ogr
import polars as pl
import geopandas as gpd


def save_households(directory: str, name: str, crs: str, output_dir: str, fraction=1.0, seed=None):
    households = pl.scan_csv(
        os.path.join(directory, f"{name}_households.csv"),
        separator=";",
        schema_overrides={
            "household_id": pl.UInt64,
            "number_of_vehicles": pl.UInt8,
            "number_of_bikes": pl.UInt8,
        },
    )
    if fraction < 1.0:
        households = households.sample(fraction=fraction, seed=seed)
    # Add number of persons in the household.
    persons = pl.scan_csv(
        os.path.join(directory, f"{name}_persons.csv"),
        separator=";",
        schema_overrides={"household_id": pl.UInt64},
    )
    household_sizes = persons.group_by("household_id").agg(pl.len().alias("number_of_persons"))
    households = households.join(household_sizes, on="household_id", how="inner")
    # Add home coordinates.
    home_coords_df = get_home_coordinates(directory, name)
    households = households.join(home_coords_df.lazy(), on="household_id", how="inner")
    # Collect and return the DataFrame.
    print("Collecting household data...")
    households = households.select(
        "household_id",
        "number_of_vehicles",
        "number_of_bikes",
        "income",
        "number_of_persons",
        "home_x",
        "home_y",
    ).collect()
    # Create a GeoDataFrame.
    gdf = gpd.GeoDataFrame(
        data=households.drop("home_x", "home_y").to_pandas(),
        geometry=gpd.points_from_xy(households["home_x"], households["home_y"], crs=crs),
    )
    n = len(gdf)
    print(f"{n:,} households collected")
    gdf.to_parquet(os.path.join(output_dir, "households.parquet"))
    return households


def get_home_coordinates(directory: str, name: str):
    print("Reading home coordinates...")
    source = ogr.Open(os.path.join(directory, f"{name}_homes.gpkg"))
    layer = source.GetLayer()
    home_coords = list()
    for feature in layer:
        (x, y, _) = feature.geometry().GetPoint()
        home_coords.append((feature["household_id"], x, y))
    home_coords_df = pl.DataFrame(
        data=home_coords,
        schema={"household_id": pl.UInt64, "home_x": pl.Float64, "home_y": pl.Float64},
    )
    return home_coords_df


def save_persons(directory: str, name: str, output_dir: str, households: pl.DataFrame):
    persons = pl.scan_csv(
        os.path.join(directory, f"{name}_persons.csv"),
        separator=";",
        schema_overrides={
            "person_id": pl.UInt64,
            "household_id": pl.UInt64,
            "age": pl.UInt8,
            "employed": pl.Boolean,
            "sex": pl.String,
            "socioprofessional_class": pl.UInt8,
            "has_driving_license": pl.Boolean,
            "has_pt_subscription": pl.Boolean,
        },
    )
    # Filter the persons with a valid household.
    persons = persons.join(households, on="household_id", how="semi")
    # Set `sex` column as a boolean.
    persons = persons.with_columns((pl.col("sex") == "female").alias("woman"))
    # Collect and return the DataFrame.
    print("Collecting person data...")
    persons = persons.select(
        "person_id",
        "household_id",
        "age",
        "employed",
        "woman",
        "socioprofessional_class",
        "has_driving_license",
        "has_pt_subscription",
    ).collect()
    n = len(persons)
    print(f"{n:,} persons collected")
    persons.write_parquet(os.path.join(output_dir, "persons.parquet"))
    return persons


def save_trips(directory: str, name: str, identify_tours: bool, crs: str, output_dir: str, persons: pl.DataFrame):
    print("Reading trips")
    trips = pl.scan_csv(
        os.path.join(directory, f"{name}_trips.csv"),
        separator=";",
        schema_overrides={
            "person_id": pl.UInt64,
            "trip_index": pl.UInt64,
            "departure_time": pl.Float64,
            "arrival_time": pl.Float64,
            "preceding_purpose": pl.String,
            "following_purpose": pl.String,
        },
    )
    # Filter trips of valid persons.
    trips = trips.filter(persons, on="person_id", how="semi")
    trips = trips.sort(["person_id", "trip_index"])
    # Collect and return the DataFrame.
    print("Collecting trip data...")
    trips = trips.select(
        "person_id",
        "trip_index",
        "departure_time",
        "arrival_time",
        "preceding_purpose",
        "following_purpose",
    ).collect()
    n = len(trips)
    print(f"{n:,} trips collected")
    if identify_tours:
        # Filter out agents whose first trip is not from home and last trip is not to home.
        trips = trips.filter(
            pl.col("preceding_purpose").first().over("person_id") == "home",
            pl.col("following_purpose").last().over("person_id") == "home",
        )
        m = n - len(trips)
        if m > 0:
            print(f"{m:,} trips were removed because the start / end activity is not \"home\"")
        trips = trips.with_columns(
            pl.when(pl.col("preceding_purpose") == "home").then(
                pl.lit(1)
            ).otherwise(None).cast(pl.UInt64).cum_count().alias("tour_id")
        )
        print("Number of tours: {:,}".format(trips["tour_id"].n_unique()))
    trips = trips.with_columns(pl.int_range(len(trips), dtype=pl.UInt64).alias("trip_id"))
    # Retrieve origin / destination coordinates.
    orig_df, dest_df = get_trip_coordinates(directory, name, crs, trips)
    # Add origin / destination latitude / longitude to the DataFrame.
    orig_df.to_crs("EPSG:4326", inplace=True)
    trips = trips.join(
        pl.DataFrame(
            {
                "trip_id": pl.from_pandas(orig_df["trip_id"]),
                "origin_lng": pl.from_pandas(orig_df.geometry.x),
                "origin_lat": pl.from_pandas(orig_df.geometry.y),
            }
        ),
        on="trip_id",
        how="left",
    )
    orig_df.to_crs(crs, inplace=True)
    dest_df.to_crs("EPSG:4326", inplace=True)
    trips = trips.join(
        pl.DataFrame(
            {
                "trip_id": pl.from_pandas(dest_df["trip_id"]),
                "destination_lng": pl.from_pandas(dest_df.geometry.x),
                "destination_lat": pl.from_pandas(dest_df.geometry.y),
            }
        ),
        on="trip_id",
        how="left",
    )
    dest_df.to_crs(crs, inplace=True)
    # Compute origin / destination distance.
    distances = orig_df.distance(dest_df) / 1e3
    trips = trips.join(
        pl.DataFrame({"trip_id": orig_df["trip_id"], "od_distance": distances.values}),
        on="trip_id",
        how="left",
    )
    trips.write_parquet(os.path.join(output_dir, "trips.parquet"))
    orig_df.to_parquet(os.path.join(output_dir, "trip_origins.parquet"))
    dest_df.to_parquet(os.path.join(output_dir, "trip_destinations.parquet"))
    return trips


def get_trip_coordinates(directory: str, name: str, crs: str, trips: pl.DataFrame):
    print("Reading trip origin / destination coordinates")
    source = ogr.Open(os.path.join(directory, f"{name}_trips.gpkg"))
    layer = source.GetLayer()
    trip_indices = list()
    origin_xs = list()
    origin_ys = list()
    destination_xs = list()
    destination_ys = list()
    idx = {(p, t) for p, t in zip(trips["person_id"], trips["trip_index"])}
    for feature in layer:
        if not (feature["person_id"], feature["trip_index"]) in idx:
            continue
        ((x0, y0), (x1, y1)) = feature.geometry().GetPoints()
        trip_indices.append((feature["person_id"], feature["trip_index"]))
        origin_xs.append(x0)
        origin_ys.append(y0)
        destination_xs.append(x1)
        destination_ys.append(y1)
    trip_indices_df = pl.DataFrame(
        data=trip_indices,
        schema={"person_id": pl.UInt64, "trip_index": pl.UInt64},
    )
    trip_ids = trip_indices_df.join(
        trips.select("person_id", "trip_index", "trip_id"),
        on=["person_id", "trip_index"],
        how="inner",
    ).select("trip_id")
    origin_coords_df = gpd.GeoDataFrame(
        data=trip_ids.to_pandas(),
        geometry=gpd.points_from_xy(origin_xs, origin_ys, crs=crs),
    )
    destination_coords_df = gpd.GeoDataFrame(
        data=trip_ids.to_pandas(),
        geometry=gpd.points_from_xy(destination_xs, destination_ys, crs=crs),
    )
    assert len(trips) == len(origin_coords_df), "Some trips do not have an origin"
    assert len(trips) == len(destination_coords_df), "Some trips do not have a destination"
    return origin_coords_df, destination_coords_df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "crs",
        "synthetic_population.input_directory",
        "synthetic_population.name",
        "population_directory",
    ]
    check_keys(config, mandatory_keys)
    directory = config["synthetic_population"]["input_directory"]
    name = config["synthetic_population"]["name"]
    identify_tours = config["synthetic_population"].get("identify_tours", False)
    output_dir = config["population_directory"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    t0 = time.time()
    households = save_households(directory, name, config["crs"], output_dir,
                                 config["synthetic_population"].get("fraction", 1.0),
                                 config.get("random_seed"))
    persons = save_persons(directory, name, output_dir, households)
    save_trips(directory, name, identify_tours, config["crs"], output_dir, persons)
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
