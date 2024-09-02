import os
import json

import polars as pl

import metropy.run.base as metro_run


def generate_agents(trips: pl.LazyFrame):
    print("Creating agent-level DataFrame")
    # Agent-level values: only `agent_id` is useful.
    agents = trips.select(pl.col("agent_id").unique()).collect()

    print("Creating alternative-level DataFrame")
    # Alternative-level values.
    alts = (
        trips.group_by("agent_id")
        .agg(
            pl.lit(1).alias("alt_id"),
            pl.col("access_time").first().alias("origin_delay"),
            pl.lit("Constant").alias("dt_choice.type"),
            pl.col("departure_time").first().alias("dt_choice.departure_time"),
        )
        .collect()
    )

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


def write_parameters(run_directory: str, config: dict):
    parameters = metro_run.PARAMETERS.copy()
    parameters["learning_model"]["value"] = config["car_only"]["smoothing_factor"]
    parameters["max_iterations"] = config["car_only"]["nb_iterations"]
    parameters["period"] = config["period"]
    parameters["road_network"]["recording_interval"] = config["recording_interval"]
    parameters["road_network"]["spillback"] = config["spillback"]
    parameters["road_network"]["max_pending_duration"] = config["max_pending_duration"]
    if "backward_wave_speed" in config:
        parameters["road_network"]["backward_wave_speed"] = config["backward_wave_speed"]
    parameters["road_network"]["algorithm_type"] = config["routing_algorithm"]
    print("Writing parameters")
    with open(os.path.join(run_directory, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "population_directory",
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

    edges = metro_run.read_edges(
        config["clean_edges_file"],
        config.get("routing", dict).get("car_split", dict).get("main_edges_filename"),
        config["edge_penalties_file"],
    )
    trips = metro_run.read_trips(
        config["population_directory"],
        config["routing"]["car_split"]["trips_filename"],
        config["run"]["period"],
    )

    agents, alts, trips, used_nodes = generate_agents(trips)

    metro_run.write_agents(run_directory, agents, alts, trips)

    edges = metro_run.generate_edges(edges, config["run"])
    vehicles = metro_run.generate_vehicles()

    all_nodes = set(edges["source"]).union(set(edges["target"]))
    if any(n not in all_nodes for n in used_nodes):
        print(
            "Warning: the origin / destination node of some trips is not a valid node of the road network"
        )

    metro_run.write_road_network(run_directory, edges, vehicles)

    write_parameters(run_directory, config["run"])
