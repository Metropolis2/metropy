# Base functions used by multiple scripts to generate simulation input.
import os

import polars as pl

import metropy.utils.io as metro_io


# Base parameters, common to all simulations.
PARAMETERS = {
    "input_files": {
        "agents": "input/agents.parquet",
        "alternatives": "input/alts.parquet",
        "trips": "input/trips.parquet",
        "edges": "input/edges.parquet",
        "vehicle_types": "input/vehicles.parquet",
    },
    "output_directory": "output",
    "learning_model": {
        "type": "Exponential",
    },
    "road_network": dict(),
    "saving_format": "Parquet",
}


def read_edges(
    edge_filename: str, edge_main_filename: None | str, edge_penalties_filename: None | str
):
    print("Reading edges")
    edges = metro_io.scan_dataframe(edge_filename)
    if edge_main_filename is not None:
        edges_main = metro_io.scan_dataframe(edge_main_filename).filter("main")
        edges = edges.join(edges_main, on=pl.col("edge_id"), how="semi")
    if edge_penalties_filename is not None:
        edges_penalties = metro_io.scan_dataframe(edge_penalties_filename).rename(
            {"additive_penalty": "constant_travel_time"}
        )
        columns = edges_penalties.collect_schema().names()
        assert (
            "constant_travel_time" in columns
        ), "No column `additive_penalty` in the edges penalties file"
        if "speed" in columns:
            edges = edges.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("constant_travel_time").fill_null(pl.lit(0.0)),
                pl.col("speed").fill_null(pl.col("speed_limit")),
            )
        else:
            print("Warning: no `speed` column in the edges penalties, using additive penalty only")
            edges = edges.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("constant_travel_time").fill_null(pl.lit(0.0)),
                pl.col("speed_limit").alias("speed"),
            )
    else:
        edges = edges.with_columns(constant_travel_time=pl.lit(0.0), speed=pl.col("speed_limit"))
    return edges


def generate_vehicles(config: dict):
    print("Creating vehicle types...")
    vehicles = [
        {
            "vehicle_id": 1,
            "headway": config.get("car_headway", 10.0) / config.get("simulation_ratio", 1.0),
            "pce": config.get("car_pce", 1.0) / config.get("simulation_ratio", 1.0),
        },
    ]
    if "truck_headway" in config and "truck_pce" in config:
        truck = {
            "vehicle_id": 2,
            "headway": config["truck_headway"] / config.get("truck_simulation_ratio", 1.0),
            "pce": config["truck_pce"] / config.get("truck_simulation_ratio", 1.0),
        }
        if "truck_speed_limit" in config:
            truck["speed_function.type"] = "UpperBound"
            truck["speed_function.upper_bound"] = config["truck_speed_limit"] / 3.6
        vehicles.append(truck)
    vehicles = pl.DataFrame(vehicles)
    return vehicles


def generate_edges(edges: pl.LazyFrame, config: dict):
    print("Creating METROPOLIS edges...")
    # Convert edges' speed from km/h to m/s.
    edges = edges.with_columns(pl.col("speed") / 3.6)
    edges = edges.with_columns(pl.lit(config.get("overtaking", True)).alias("overtaking"))
    columns = [
        "edge_id",
        "source",
        "target",
        "speed",
        "length",
        "lanes",
        "overtaking",
        "constant_travel_time",
    ]
    sort_columns = ["lanes", "speed", "length"]
    sort_descending = [True, True, False]
    if config.get("use_bottleneck", False):
        assert isinstance(
            config["edge_capacity"], dict
        ), "Edge capacities must be specified when `use_bottleneck=true`."
        edges = edges.with_columns(
            bottleneck_flow=pl.col("road_type").replace_strict(
                config["edge_capacity"], return_dtype=pl.Float64
            )
            / 3600
        )
        columns.append("bottleneck_flow")
        sort_columns.insert(0, "bottleneck_flow")
        sort_descending.insert(0, True)
    edges_df = edges.select(columns).collect()
    # Remove parallel edges.
    n0 = len(edges_df)
    edges_df = edges_df.sort(sort_columns, descending=sort_descending).unique(
        subset=["source", "target"], keep="first"
    )
    n1 = len(edges_df)
    if n0 > n1:
        print("Warning: Discarded {:,} parallel edges".format(n0 - n1))
    edges_df = edges_df.sort("source")
    return edges_df


def read_trips(
    population_directory: str, road_split_filename: str, period: list[float], road_only=False
):
    print("Reading trips")
    trips = (
        metro_io.scan_dataframe(os.path.join(population_directory, "trips.parquet"))
        .rename({"person_id": "agent_id"})
        .with_columns(is_truck=False)
    )
    if road_only:
        trip_modes = metro_io.scan_dataframe(
            os.path.join(population_directory, "trip_modes.parquet")
        ).filter(pl.col("mode") == "car_driver")
        trips = trips.join(trip_modes, on="trip_id", how="semi")
    truck_filename = os.path.join(population_directory, "truck_trips.parquet")
    if os.path.isfile(truck_filename):
        truck_trips = metro_io.scan_dataframe(truck_filename)
        trips = pl.concat(
            (trips, truck_trips.with_columns(is_truck=True)),
            how="diagonal_relaxed",
        )
    # Remove trips with origin = destination.
    trips = trips.filter(
        (pl.col("origin_lng") != pl.col("destination_lng"))
        | (pl.col("origin_lat") != pl.col("destination_lat"))
    )
    # Remove trips outside of the simulation period.
    # NOTE. The format below assume that times are repeat each 24h. So if the period is 6h to 25h,
    # then all departure times that are smaller than 1h or larger than 6h, after modulo 24h, will be
    # considered.
    trips = trips.filter(
        ((pl.col("departure_time") - period[0]) % (24.0 * 3600.0)).is_between(
            0.0, period[1] - period[0]
        )
    )
    road_split = metro_io.scan_dataframe(road_split_filename)
    trips = trips.join(road_split, on="trip_id", how="left", coalesce=True)
    return trips


def write_agents(run_directory: str, agents: pl.DataFrame, alts: pl.DataFrame, trips: pl.DataFrame):
    print("Writing agents")
    agents.write_parquet(os.path.join(run_directory, "input", "agents.parquet"), use_pyarrow=True)
    print("Writing alternatives")
    alts.write_parquet(os.path.join(run_directory, "input", "alts.parquet"), use_pyarrow=True)
    print("Writing trips")
    trips.write_parquet(os.path.join(run_directory, "input", "trips.parquet"), use_pyarrow=True)


def write_road_network(run_directory: str, edges: pl.DataFrame, vehicles: pl.DataFrame):
    print("Writing edges")
    edges.write_parquet(os.path.join(run_directory, "input", "edges.parquet"), use_pyarrow=True)
    print("Writing vehicle types")
    vehicles.write_parquet(
        os.path.join(run_directory, "input", "vehicles.parquet"), use_pyarrow=True
    )
