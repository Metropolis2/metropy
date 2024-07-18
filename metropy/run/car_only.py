import os
import json

import polars as pl

import metropy.utils.io as metro_io


# Parameters to use for the simulation.
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
            # Drop the speed column and join to replace it with the penalty speed.
            edges = edges.drop("speed").join(edges_penalties, on=pl.col("edge_id"))
        else:
            print("Warning: no `speed` column in the edges penalties, using additive penalty only")
    else:
        edges = edges.with_columns(constant_travel_time=pl.lit(0.0))
    return edges


def generate_road_network(edges: pl.LazyFrame, config: dict):
    print("Creating Metropolis road network")
    vehicles = [
        {
            "vehicle_id": 1,
            "headway": config.get("car_headway", 10.0) / config.get("simulation_ratio", 1.0),
            "pce": config.get("car_pce", 1.0) / config.get("simulation_ratio", 1.0),
        },
    ]
    vehicles = pl.DataFrame(vehicles)
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
            bottleneck_flow=pl.col("road_type").replace_strict(config["edge_capacity"], return_dtype=pl.Float64)
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
    return edges_df, vehicles


def read_trips(population_directory: str, car_split_filename: str):
    print("Reading trips")
    trips = metro_io.scan_dataframe(os.path.join(population_directory, "trips.parquet")).rename(
        {"person_id": "agent_id"}
    )
    # Remove trips with origin = destination.
    trips = trips.filter(
        (pl.col("origin_lng") != pl.col("destination_lng"))
        | (pl.col("origin_lat") != pl.col("destination_lat"))
    )
    car_split = metro_io.scan_dataframe(car_split_filename)
    trips = trips.join(car_split, on="trip_id", how="left", coalesce=True)
    trip_modes = metro_io.scan_dataframe(
        os.path.join(population_directory, "trip_modes.parquet")
    ).filter(pl.col("mode") == "car_driver")
    trips = trips.join(trip_modes, on="trip_id", how="semi")
    trips = trips.sort("agent_id")
    return trips


def generate_agents(trips: pl.LazyFrame):
    print("Creating agent-level DataFrame")
    # Agent-level values: only `agent_id` is useful.
    agents = trips.select(pl.col("agent_id").unique()).collect()

    print("Creating alternative-level DataFrame")
    # Alternative-level values.
    alts = trips.group_by("agent_id").agg(
        pl.lit(1).alias("alt_id"),
        pl.col("access_time").first().alias("origin_delay"),
        pl.lit("Constant").alias("dt_choice.type"),
        pl.col("departure_time").first().alias("dt_choice.departure_time"),
    ).collect()

    print("Creating trip-level DataFrame")
    # Trip-level values.
    trips_df = trips.select(
        pl.col("agent_id"),
        pl.lit(1).alias("alt_id"),
        pl.col("trip_id"),
        # Stopping time is actual stopping time + egress time + access time of next trip.
        (
            (
                pl.col("departure_time").shift(-1).over("agent_id") - pl.col("arrival_time")
            ).fill_null(0.0)
            + pl.col("egress_time")
            + (pl.col("access_time").shift(-1, fill_value=0.0).over("agent_id"))
        ).alias("stopping_time"),
        # Trip is virtual for trips only on the secondary network.
        pl.when(pl.col("secondary_only"))
        .then(pl.lit("Virtual"))
        .otherwise(pl.lit("Road"))
        .alias("class.type"),
        pl.when(pl.col("secondary_only"))
        .then(pl.col("free_flow_travel_time"))
        .otherwise(pl.lit(None))
        .alias("class.travel_time"),
        pl.when(pl.col("secondary_only"))
        .then(pl.lit(None))
        .otherwise(pl.col("access_node"))
        .alias("class.origin"),
        pl.when(pl.col("secondary_only"))
        .then(pl.lit(None))
        .otherwise(pl.col("egress_node"))
        .alias("class.destination"),
        pl.when(pl.col("secondary_only"))
        .then(pl.lit(None))
        .otherwise(pl.lit(1))
        .alias("class.vehicle"),
    ).collect()

    print(
        "Generated {:,} agents, with {:,} alternatives and {:,} trips ({:,} being road trips)".format(
            agents.height,
            alts.height,
            trips_df.height,
            trips_df.select(pl.col("class.type") == "Road").sum().item(),
        )
    )
    nb_origins = trips_df["class.origin"].n_unique()
    print("Number of unique origins: {}".format(nb_origins))
    nb_destinations = trips_df["class.destination"].n_unique()
    print("Number of unique destinations: {}".format(nb_destinations))
    used_nodes = set(trips_df["class.origin"]).union(set(trips_df["class.destination"]))
    used_nodes.discard(None)
    return agents, alts, trips_df, used_nodes


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
    vehicles.write_parquet(os.path.join(run_directory, "input", "vehicles.parquet"), use_pyarrow=True)


def write_parameters(run_directory: str, config: dict):
    PARAMETERS["learning_model"]["value"] = config["car_only"]["smoothing_factor"]
    PARAMETERS["max_iterations"] = config["car_only"]["nb_iterations"]
    PARAMETERS["period"] = config["period"]
    PARAMETERS["road_network"]["recording_interval"] = config["recording_interval"]
    PARAMETERS["road_network"]["spillback"] = config["spillback"]
    PARAMETERS["road_network"]["max_pending_duration"] = config["max_pending_duration"]
    PARAMETERS["road_network"]["algorithm_type"] = config["routing_algorithm"]
    print("Writing parameters")
    with open(os.path.join(run_directory, "parameters.json"), "w") as f:
        f.write(json.dumps(PARAMETERS))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "population_directory",
        "routing.car_split.main_edges_filename",
        "run.edge_capacity",
        "run.period",
        "run.recording_interval",
        "run.spillback",
        "run.max_pending_duration",
        "run.routing_algorithm",
        "run.car_only.directory",
        "run.car_only.smoothing_factor",
        "run.car_only.nb_iterations",
    ]
    check_keys(config, mandatory_keys)

    run_directory = config["run"]["car_only"]["directory"]
    if not os.path.isdir(os.path.join(run_directory, "input")):
        os.makedirs(os.path.join(run_directory, "input"))

    edges = read_edges(config["clean_edges_file"],
                       config["routing"]["car_split"]["main_edges_filename"],
                       config["edge_penalties_file"])
    trips = read_trips(config["population_directory"], config["car_split"]["trips_filename"])

    agents, alts, trips, used_nodes = generate_agents(trips)

    write_agents(run_directory, agents, alts, trips)

    edges, vehicles = generate_road_network(edges, config["run"])

    all_nodes = set(edges["source"]).union(set(edges["target"]))
    if any(n not in all_nodes for n in used_nodes):
        print(
            "Warning: the origin / destination node of some trips is not a valid node of the road network"
        )

    write_road_network(run_directory, edges, vehicles)

    write_parameters(run_directory, config["run"])
