import os
import json

import polars as pl

import metropy.utils.io as metro_io
import metropy.run.base as metro_run


def read_tomtom_paths(input_file: str, edges: pl.DataFrame, period: list[float]):
    print("Reading TomTom paths...")
    lf = metro_io.scan_dataframe(input_file)
    # Filter-out trips during the weekends.
    lf = lf.filter(pl.col("departure_time").dt.weekday() <= 5)
    # Create departure time column (in seconds after midnight).
    lf = lf.with_columns(
        dt=(
            pl.col("departure_time").dt.hour().cast(pl.UInt64) * 3600
            + pl.col("departure_time").dt.minute().cast(pl.UInt64) * 60
            + pl.col("departure_time").dt.second().cast(pl.UInt64)
        ).cast(pl.Float64)
    )
    # Filter-out trips outside of the period.
    lf = lf.filter(pl.col("dt") >= period[0], pl.col("dt") <= period[1])
    # Filter-out paths where some edges are not part of the road network (probably the parallel
    # edges).
    edge_ids = set(edges["edge_id"])
    lf = lf.filter(pl.col("cpath").list.eval(pl.element().is_in(edge_ids)).list.all())
    # Find origin and destination nodes.
    lf = lf.with_columns(
        pl.col("cpath")
        .list.first()
        .replace_strict(edges["edge_id"], edges["source"])
        .alias("origin"),
        pl.col("cpath")
        .list.last()
        .replace_strict(edges["edge_id"], edges["target"])
        .alias("destination"),
    )
    df = lf.select("id", "cpath", "dt", "origin", "destination").collect()
    print("Number of paths read: {:,}".format(len(df)))
    return df


def generate_agents(tomtom: pl.DataFrame):
    tomtom = tomtom.with_columns(agent_id=pl.col("id"))
    assert len(tomtom) == tomtom["agent_id"].n_unique(), "Id should be unique over TomTom requests"
    print("Creating agent-level DataFrame")
    # Agent-level values: only `agent_id` is useful.
    agents = tomtom.select("agent_id")

    print("Creating alternative-level DataFrame")
    # Alternative-level values.
    alts = tomtom.select(
        "agent_id",
        pl.lit(1).alias("alt_id"),
        pl.lit("Constant").alias("dt_choice.type"),
        pl.col("dt").alias("dt_choice.departure_time"),
    )

    print("Creating trip-level DataFrame")
    # Trip-level values.
    trips = tomtom.select(
        "agent_id",
        pl.lit(1).alias("alt_id"),
        pl.col("agent_id").alias("trip_id"),
        pl.lit("Road").alias("class.type"),
        pl.col("origin").alias("class.origin"),
        pl.col("destination").alias("class.destination"),
        pl.lit(1).alias("class.vehicle"),
        pl.col("cpath").alias("class.route"),
    )

    print(
        "Generated {:,} agents, with {:,} alternatives and {:,} trips".format(
            agents.height,
            alts.height,
            trips.height,
        )
    )
    nb_origins = trips["class.origin"].n_unique()
    print("Number of unique origins: {}".format(nb_origins))
    nb_destinations = trips["class.destination"].n_unique()
    print("Number of unique destinations: {}".format(nb_destinations))
    return agents, alts, trips


def write_parameters(run_directory: str, config: dict, road_only_directory: str):
    parameters = metro_run.PARAMETERS.copy()
    parameters["learning_model"]["value"] = 0.0
    parameters["input_files"]["road_network_conditions"] = os.path.join(
        os.path.abspath(road_only_directory), "output", "net_cond_sim_edge_ttfs.parquet"
    )
    parameters["period"] = config["period"]
    parameters["road_network"]["recording_interval"] = config["recording_interval"]
    parameters["road_network"]["spillback"] = config["spillback"]
    parameters["road_network"]["max_pending_duration"] = config["max_pending_duration"]
    if "backward_wave_speed" in config:
        parameters["road_network"]["backward_wave_speed"] = config["backward_wave_speed"]
    parameters["road_network"]["algorithm_type"] = config["routing_algorithm"]
    parameters["only_compute_decisions"] = True
    print("Writing parameters")
    with open(os.path.join(run_directory, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "calibration.post_map_matching.output_filename",
        "run.period",
        "run.recording_interval",
        "run.spillback",
        "run.max_pending_duration",
        "run.routing_algorithm",
        "run.road_only.directory",
        "run.tomtom_routes.directory",
    ]
    check_keys(config, mandatory_keys)

    run_directory = config["run"]["tomtom_routes"]["directory"]
    if not os.path.isdir(os.path.join(run_directory, "input")):
        os.makedirs(os.path.join(run_directory, "input"))

    # Read edges without the main file so that all edges are included (except the parallel ones).
    edges = metro_run.read_edges(
        config["clean_edges_file"],
        None,
        config.get("calibration", dict())
        .get("free_flow_calibration", dict())
        .get("output_filename"),
        config.get("capacities_filename"),
    )
    edges = metro_run.generate_edges(edges, config["run"])
    vehicles = metro_run.generate_vehicles(config["run"])

    tomtom = read_tomtom_paths(
        config["calibration"]["post_map_matching"]["output_filename"],
        edges,
        config["run"]["period"],
    )
    agents, alts, trips = generate_agents(tomtom)

    metro_run.write_agents(run_directory, agents, alts, trips)

    metro_run.write_road_network(run_directory, edges, vehicles)

    write_parameters(run_directory, config["run"], config["run"]["road_only"]["directory"])
