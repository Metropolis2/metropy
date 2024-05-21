import os
import time

import polars as pl
import folium
from folium.vector_layers import Circle, PolyLine
from matplotlib import colormaps
from matplotlib.colors import to_hex


def read_household(population_directory: str, household_id: int):
    households = pl.scan_parquet(os.path.join(population_directory, "households.parquet"))
    persons = pl.scan_parquet(os.path.join(population_directory, "persons.parquet"))
    trips = pl.scan_parquet(os.path.join(population_directory, "trips.parquet"))
    households = households.filter(pl.col("household_id") == household_id)
    df = households.join(persons, on="household_id").collect()
    assert not df.is_empty()
    nb_vehicles = df["number_of_vehicles"][0]
    nb_bikes = df["number_of_bikes"][0]
    income = df["income"][0]
    print(f"Household id: {household_id}")
    print(f"Number of vehicles: {nb_vehicles}")
    print(f"Number of bikes: {nb_bikes}")
    print(f"Income: {income}")
    for i, row in enumerate(df.iter_rows(named=True)):
        print(f"=== Person {i + 1} ===")
        print(f"Id: {row['person_id']}")
        print(f"Age: {row['age']}")
        if row["woman"]:
            print("Woman")
        else:
            print("Man")
        print(f"Employed: {row['employed']}")
        print(f"CSP: {row['socioprofessional_class']}")
    df = df.lazy().join(trips, on="person_id").collect()
    return df


def create_household_map(df: pl.DataFrame):
    mean_location = (df["origin_lat"].mean(), df["origin_lng"].mean())
    m = folium.Map(
        location=mean_location,
        zoom_start=13,
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='\u003ca href="https://www.openstreetmap.org/copyright" target="_blank"\u003e\u0026copy; OpenStreetMap contributors\u003c/a\u003e',
    )
    cmp = colormaps["Set3"]
    colors = {person_id: to_hex(cmp(i)) for i, person_id in enumerate(df["person_id"].unique())}
    for i, row in enumerate(df.iter_rows(named=True)):
        td = get_time_str(row["departure_time"])
        ta = get_time_str(row["arrival_time"])
        print(f"=== Trip {i + 1} ===")
        print(f"Person id: {row['person_id']}")
        print(f"Trip id: {row['trip_id']}")
        print(f"Departure time: {td}")
        print(f"Arrival time: {ta}")
        print(f"Preceding purpose: {row['preceding_purpose']}")
        print(f"Following purpose: {row['following_purpose']}")
        msg = f"""
        Origin of trip {row['trip_id']} of person {row['person_id']}<br>
        Purpose: {row['preceding_purpose']}<br>
        Departure time: {td}
        """
        Circle(
            location=(row["origin_lat"], row["origin_lng"]),
            radius=20,
            tooltip=msg,
            opacity=0.5,
            fill_opacity=0.5,
            color=colors[row["person_id"]],
            fill_color=colors[row["person_id"]],
        ).add_to(m)
        msg = f"""
        Destination of trip {row['trip_id']} of person {row['person_id']}<br>
        Purpose: {row['following_purpose']}<br>
        Arrival time: {ta}
        """
        Circle(
            location=(row["destination_lat"], row["destination_lng"]),
            radius=20,
            tooltip=msg,
            opacity=0.5,
            fill_opacity=0.5,
            color=colors[row["person_id"]],
            fill_color=colors[row["person_id"]],
        ).add_to(m)
        msg = f"""
        Trip {row['trip_id']} of person {row['person_id']}<br>
        Purpose: {row['preceding_purpose']} -> {row['following_purpose']}<br>
        Time: {td} to {ta}
        """
        PolyLine(
            locations=(
                (row["origin_lat"], row["origin_lng"]),
                (row["destination_lat"], row["destination_lng"]),
            ),
            tooltip=msg,
            opacity=0.5,
            color=colors[row["person_id"]],
            weight=10,
        ).add_to(m)
    return m


def get_time_str(seconds_after_midnight):
    t = round(seconds_after_midnight)
    hours = t // 3600
    remainder = t % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "synthetic_population.plot.household_id",
        "synthetic_population.plot.filename",
    ]
    check_keys(config, mandatory_keys)
    directory = config["population_directory"]

    t0 = time.time()
    df = read_household(directory, config["synthetic_population"]["plot"]["household_id"])
    m = create_household_map(df)
    m.save(config["synthetic_population"]["plot"]["filename"])
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
