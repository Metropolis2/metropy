import os
import json

import numpy as np
import polars as pl

import metropy.utils.io as metro_io
import metropy.run.base as metro_run

DT_PARAMETERS = {
    "mu": 0.96,
    "beta": {
        "work": 0.61,
        "education": 0.01,
        "shop": 0.85,
        "leisure": 0.25,
        "other": 0.66,
    },
    "gamma": {
        "work": 1.16,
        "education": 0.64,
        "shop": 0.63,
        "leisure": 0.03,
        "other": 1.08,
    },
}


def process_trips(
    trips: pl.LazyFrame,
    population_dir: str,
    walking_filename: str | None,
    pt_filename: str | None,
    fuel_filename: str | None,
    modes: list[str],
):
    # Add desired activity start time and duration.
    desired_times_filename = os.path.join(population_dir, "desired_times.parquet")
    if os.path.isfile(desired_times_filename):
        desired_times = metro_io.scan_dataframe(desired_times_filename)
        trips = trips.join(desired_times, on="trip_id", how="left")
        trips = trips.with_columns(
            pl.when(pl.col("start_time").is_not_null())
            .then("start_time")
            .otherwise("arrival_time")
            .alias("activity_start_time"),
            pl.when(pl.col("duration").is_not_null())
            .then("duration")
            .otherwise("activity_time")
            .alias("activity_duration"),
        )
    else:
        trips = trips.with_columns(
            pl.col("arrival_time").alias("activity_start_time"),
            pl.col("activity_time").alias("activity_duration"),
        )
    # Add walking distances.
    if walking_filename is not None:
        df = metro_io.scan_dataframe(walking_filename)
        trips = trips.join(df, on="trip_id", how="left").rename({"distance": "walking_distance"})
    # Add public-transit travel times.
    if pt_filename is not None:
        df = metro_io.scan_dataframe(pt_filename)
        trips = trips.join(df, on="trip_id", how="left").rename(
            {"travel_time": "public_transit_travel_time"}
        )
    # Add fuel consumption.
    if fuel_filename is not None:
        df = metro_io.scan_dataframe(fuel_filename)
        trips = trips.join(df, on="trip_id", how="left")
    # Add socio-economic characteristics.
    persons = metro_io.scan_dataframe(os.path.join(population_dir, "persons.parquet"))
    trips = trips.join(persons, on="person_id")
    # Add utility parameters.
    parameters = list()
    for woman in (False, True):
        for spclass in (
            (3,),
            (4,),
            (5,),
            (
                1,
                2,
                6,
            ),
            (7,),
            (8,),
        ):
            if woman:
                suffix = "female"
            else:
                suffix = "male"
            suffix += str(spclass[-1])
            with open(
                os.path.join("./output/calibration/", f"mode_parameters_{suffix}.json"), "r"
            ) as f:
                values = json.load(f)
            df = (
                pl.DataFrame(
                    {
                        "mode": values["cst"].keys(),
                        "cst": values["cst"].values(),
                        "vot": values["vot"].values(),
                    }
                )
                .with_columns(
                    woman=pl.lit(woman),
                    socioprofessional_class=pl.lit(spclass),
                )
                .pivot(on="mode", index=["woman", "socioprofessional_class"], values=["cst", "vot"])
                .with_columns(
                    fuel_factor=pl.lit(values["fuel_factor"]),
                )
                .explode("socioprofessional_class")
                .with_columns(pl.col("socioprofessional_class").cast(pl.UInt8))
            )
            parameters.append(df)
    parameters = pl.concat(parameters, how="vertical")
    trips = trips.join(parameters.lazy(), on=["woman", "socioprofessional_class"], how="left")
    # Add household characteristics.
    households = metro_io.scan_dataframe(os.path.join(population_dir, "households.parquet"))
    trips = trips.join(households, on="household_id")
    trips = trips.with_columns(has_car=pl.col("number_of_vehicles") > 0)
    trips = trips.select(
        "agent_id",
        "person_id",
        "trip_id",
        "walk_only",
        "preceding_purpose",
        "following_purpose",
        "od_distance",
        "tour_id",
        "access_node",
        "egress_node",
        "access_time",
        "egress_time",
        "free_flow_travel_time",
        "secondary_only",
        "activity_start_time",
        "activity_duration",
        "walking_distance",
        "public_transit_travel_time",
        "fuel_consumption",
        "has_car",
        "has_driving_license",
        *[f"cst_{mode}" for mode in modes],
        *[f"vot_{mode}" for mode in modes],
        "fuel_factor",
    )
    return trips.collect()


def split_trips(trips: pl.LazyFrame):
    truck_trips = (
        trips.filter("is_truck")
        .select(
            "agent_id",
            "trip_id",
            "access_node",
            "egress_node",
            "departure_time",
            "secondary_only",
        )
        .collect()
    )
    trips = trips.filter(pl.col("is_truck").not_())
    return trips, truck_trips


def generate_agents(trips: pl.DataFrame, modes: list[str], run_config: dict, random_seed=None):
    rng = np.random.default_rng(random_seed)
    print("Generating agents...")
    if "tour_id" in trips.columns:
        id_col = "tour_id"
    else:
        assert "agent_id" in trips.columns
        id_col = "agent_id"

    # Trip-level values.
    trips = trips.with_columns(tour_has_no_walk_pt=pl.col("walk_only").not_().any().over(id_col))
    TT_COL = {
        "walking": pl.col("walking_distance") / (run_config["walking_speed"] / 3.6),
        "bicycle": pl.col("walking_distance") / (run_config["bicycle_speed"] / 3.6),
        "public_transit": pl.col("public_transit_travel_time"),
        "motorcycle": pl.col("free_flow_travel_time"),
    }
    AVAILABLE_COL = {
        "walking": pl.lit(True),
        "bicycle": pl.lit(True),
        "public_transit": pl.col("public_transit_travel_time")
        .is_not_null()
        .and_(pl.col("tour_has_no_walk_pt")),
        "motorcycle": pl.col("has_motorcycle"),
        "car_driver": pl.col("has_driving_license").and_(pl.col("has_car")),
        "car_passenger": pl.col("has_car"),
    }
    COMMON_COLS = [
        pl.col(id_col).alias("agent_id"),
        pl.col("trip_id"),
        pl.lit("AlphaBetaGamma").alias("schedule_utility.type"),
        pl.col("following_purpose")
        .replace_strict(DT_PARAMETERS["beta"], default=0.0)
        .alias("schedule_utility.beta"),
        pl.col("following_purpose")
        .replace_strict(DT_PARAMETERS["gamma"], default=0.0)
        .alias("schedule_utility.gamma"),
    ]
    trips_df = list()
    for i, mode in enumerate(modes):
        mode_trips = trips.filter(AVAILABLE_COL[mode].all().over(id_col))
        if mode in ("car_driver", "car_passenger"):
            mode_trips = mode_trips.select(
                pl.lit(i).alias("alt_id"),
                # Constant utility includes the VOT utility of access and egress time and the fuel
                # consumption.
                # The `fill_null` ensure that the access / egress utility is zero for secondary-only
                # trips and that `fuel_consumption` is zero for origin = destination trips.
                (
                    -pl.col(f"vot_{mode}")
                    * (pl.col("access_time").fill_null(0.0) + pl.col("egress_time").fill_null(0.0))
                    # TODO fuel cost should only be there for driver!
                    - pl.col("fuel_factor") * pl.col("fuel_consumption").fill_null(0.0)
                    - pl.col(f"cst_{mode}")
                ).alias("constant_utility"),
                -pl.col(f"vot_{mode}").alias("travel_utility.one"),
                # Stopping time needs to account for access / egress times.
                (
                    pl.col("activity_duration")
                    + pl.col("egress_time")
                    + (pl.col("access_time").shift(-1, fill_value=0.0).over(id_col))
                ).alias("stopping_time"),
                # The desired arrival time needs to account for egress time.
                (pl.col("activity_start_time") - pl.col("egress_time").fill_null(0.0)).alias(
                    "schedule_utility.tstar"
                ),
                # Trip is virtual for trips only on the secondary network.
                pl.when(pl.col("secondary_only"))
                .then(pl.lit("Virtual"))
                .otherwise(pl.lit("Road"))
                .alias("class.type"),
                pl.when(pl.col("secondary_only"))
                .then(pl.col("free_flow_travel_time"))
                .otherwise(None)
                .alias("class.travel_time"),
                pl.col("access_node").alias("class.origin"),
                pl.col("egress_node").alias("class.destination"),
                *COMMON_COLS,
            )
            if mode == "car_driver":
                mode_trips = mode_trips.with_columns(
                    pl.when(pl.col("class.travel_time").is_null())
                    .then(1)
                    .otherwise(None)
                    .alias("class.vehicle"),
                )
            else:
                mode_trips = mode_trips.with_columns(
                    pl.when(pl.col("class.travel_time").is_null())
                    .then(2)
                    .otherwise(None)
                    .alias("class.vehicle"),
                )
        else:
            mode_trips = mode_trips.select(
                pl.lit(i).alias("alt_id"),
                pl.lit("Virtual").alias("class.type"),
                TT_COL[mode].alias("class.travel_time"),
                -pl.col(f"cst_{mode}").alias("constant_utility"),
                -pl.col(f"vot_{mode}").alias("travel_utility.one"),
                pl.col("activity_duration").alias("stopping_time"),
                pl.col("activity_start_time").alias("schedule_utility.tstar"),
                *COMMON_COLS,
            )
        assert (mode_trips["travel_utility.one"] <= 0.0).all()
        trips_df.append(mode_trips)
    trips_df = pl.concat(trips_df, how="diagonal_relaxed").sort("agent_id", "alt_id", "trip_id")
    if "class.origin" in trips_df.columns and "class.travel_time" in trips_df.columns:
        assert trips_df.select(
            (pl.col("class.origin").is_not_null() | pl.col("class.travel_time").is_not_null()).all()
        ).item(), "The origin / destination is unknown for some trips"
    assert trips_df["schedule_utility.tstar"].is_null().sum() == 0
    for col in (
        "agent_id",
        "alt_id",
        "trip_id",
        "class.type",
        "constant_utility",
        "travel_utility.one",
        "schedule_utility.tstar",
        "schedule_utility.type",
        "schedule_utility.beta",
        "schedule_utility.gamma",
    ):
        assert trips_df[col].null_count() == 0, f"Found null values for column `{col}`"

    nb_tours = trips_df["agent_id"].n_unique()
    nb_alts = (
        trips_df.group_by("agent_id")
        .agg(nb_alts=pl.col("alt_id").n_unique())
        .select("nb_alts")
        .to_series()
    )
    dt_us = np.repeat(rng.random(size=nb_tours), nb_alts)
    DT_COLUMNS = [
        pl.lit("Continuous").alias("dt_choice.type"),
        pl.lit("Logit").alias("dt_choice.model.type"),
        pl.lit(DT_PARAMETERS["mu"]).alias("dt_choice.model.mu"),
        pl.Series(dt_us).alias("dt_choice.model.u"),
    ]
    alts = (
        trips_df.select("agent_id", "alt_id", "class.type")
        .unique(subset=["agent_id", "alt_id"])
        .sort("agent_id", "alt_id")
        .with_columns(
            *DT_COLUMNS,
            pl.lit(True).alias("pre_compute_route"),
        )
        .join(
            trips.group_by(id_col).agg(pl.col("access_time").first()),
            left_on=pl.col("agent_id"),
            right_on=id_col,
            how="left",
        )
        .with_columns(
            pl.when(pl.col("class.type") == "Road")
            .then("access_time")
            .otherwise(pl.lit(None))
            .alias("origin_delay")
        )
        .select(
            "agent_id",
            "alt_id",
            "pre_compute_route",
            "dt_choice.type",
            "dt_choice.model.type",
            "dt_choice.model.mu",
            "dt_choice.model.u",
            "origin_delay",
        )
    )
    # TODO check that origin delay is well implemented.
    for col in (
        "agent_id",
        "alt_id",
        "dt_choice.type",
        "dt_choice.model.type",
        "dt_choice.model.mu",
        "dt_choice.model.u",
    ):
        assert alts[col].null_count() == 0, f"Found null values for column `{col}`"

    # Agent-level values.
    nb_agents = alts["agent_id"].n_unique()
    mode_us = rng.random(size=nb_agents)
    agents = (
        trips_df.select("agent_id")
        .unique()
        .with_columns(
            pl.lit("Logit").alias("alt_choice.type"),
            pl.Series(mode_us).alias("alt_choice.u"),
            pl.lit(1.0).alias("alt_choice.mu"),
        )
    )
    for col in (
        "agent_id",
        "alt_choice.type",
        "alt_choice.u",
        "alt_choice.mu",
    ):
        assert agents[col].null_count() == 0, f"Found null values for column `{col}`"

    print(
        "Generated {:,} agents, with {:,} alternatives and {:,} trips ({:,} being road trips)".format(
            agents.height,
            alts.height,
            trips_df.height,
            trips_df.select(pl.col("class.type") == "Road").sum().item(),
        )
    )
    return agents, alts, trips_df


def generate_truck_agents(truck_trips: pl.DataFrame, congestion_run: str, edges: pl.DataFrame):
    if truck_trips.is_empty():
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    print("Generating truck agents...")
    # We don't need to simulate the secondary-only trips.
    truck_trips = truck_trips.filter(pl.col("secondary_only").not_())
    # Add routes.
    # TODO: We can only run with route choice.
    routes = (
        pl.scan_parquet(os.path.join(congestion_run, "output", "route_results.parquet"))
        .filter(pl.col("trip_id").is_in(truck_trips["trip_id"]))
        .sort("trip_id", "entry_time")
        # Filter-out secondary edges.
        .join(edges.lazy(), on="edge_id", how="semi")
        .group_by("trip_id")
        .agg(pl.col("edge_id").alias("route"))
        .collect()
    )
    truck_trips = truck_trips.join(routes, on="trip_id", how="left")
    # Trip-level values.
    trips = truck_trips.select(
        "agent_id",
        pl.lit(1).alias("alt_id"),
        "trip_id",
        pl.lit("Road").alias("class.type"),
        pl.col("access_node").alias("class.origin"),
        pl.col("egress_node").alias("class.destination"),
        pl.lit(3).alias("class.vehicle"),
        pl.col("route").alias("class.route"),
    )
    for col in (
        "agent_id",
        "alt_id",
        "trip_id",
        "class.type",
        "class.origin",
        "class.destination",
        "class.vehicle",
        "class.route",
    ):
        assert trips[col].null_count() == 0, f"Found null values for column `{col}`"

    # Alternative-level values.
    alts = truck_trips.select(
        "agent_id",
        pl.lit(1).alias("alt_id"),
        pl.lit("Constant").alias("dt_choice.type"),
        pl.col("departure_time").alias("dt_choice.departure_time"),
    )
    for col in (
        "agent_id",
        "alt_id",
        "dt_choice.type",
        "dt_choice.departure_time",
    ):
        assert alts[col].null_count() == 0, f"Found null values for column `{col}`"

    # Agent-level values.
    agents = truck_trips.select("agent_id")
    assert agents["agent_id"].null_count() == 0, f"Found null values for column `agent_id`"

    print(
        "Generated {:,} agents, with {:,} alternatives and {:,} trips ({:,} being road trips)".format(
            agents.height,
            alts.height,
            trips.height,
            trips.select(pl.col("class.type") == "Road").sum().item(),
        )
    )
    return agents, alts, trips


def merge_populations(
    agents: pl.DataFrame,
    alts: pl.DataFrame,
    trips: pl.DataFrame,
    truck_agents: pl.DataFrame,
    truck_alts: pl.DataFrame,
    truck_trips: pl.DataFrame,
):
    agents = pl.concat((agents, truck_agents), how="diagonal_relaxed")
    alts = pl.concat((alts, truck_alts), how="diagonal_relaxed")
    trips = pl.concat((trips, truck_trips), how="diagonal_relaxed")
    nodes = set(
        list(
            trips.filter(pl.col("class.origin").is_not_null())
            .select("class.origin", "class.destination")
            .to_numpy()
            .flatten()
        )
    )
    return agents, alts, trips, nodes


def write_parameters(run_directory: str, config: dict, congestion_run: str):
    parameters = metro_run.PARAMETERS.copy()
    parameters["max_iterations"] = config["nb_iterations"]
    parameters["learning_model"]["value"] = config["smoothing_factor"]
    parameters["input_files"]["road_network_conditions"] = os.path.join(
        os.path.abspath(congestion_run), "output", "net_cond_next_exp_edge_ttfs.parquet"
    )
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
        "capacities_filename",
        "population_directory",
        "run_directory",
        "run.period",
        "run.nb_iterations",
        "run.smoothing_factor",
        "run.recording_interval",
        "run.spillback",
        "run.max_pending_duration",
        "run.routing_algorithm",
        "run.road_only.directory",
        "demand.modes",
    ]
    check_keys(config, mandatory_keys)

    run_directory = config["run_directory"]
    if not os.path.isdir(os.path.join(run_directory, "input")):
        os.makedirs(os.path.join(run_directory, "input"))

    edges = metro_run.read_edges(
        config["clean_edges_file"],
        config.get("routing", dict()).get("road_split", dict()).get("main_edges_filename"),
        config.get("calibration", dict()).get("edge_penalties", dict()).get("output_filename"),
        config.get("capacities_filename"),
    )
    edges = metro_run.generate_edges(edges, config["run"])
    vehicles = metro_run.generate_vehicles(config["run"])

    trips = metro_run.read_trips(
        config["population_directory"],
        config["routing"]["road_split"]["trips_filename"],
        config["run"]["period"],
        include_trucks=True,
        road_only=False,
        modes=config["demand"]["modes"],
    )

    trips, truck_trips = split_trips(trips)

    trips = process_trips(
        trips,
        config["population_directory"],
        config["routing"].get("walking_distance", dict()).get("output_filename"),
        config["routing"].get("opentripplanner", dict()).get("output_filename"),
        config.get("fuel_consumption", dict()).get("output_filename"),
        config["demand"]["modes"],
    )

    agents, alts, trips = generate_agents(
        trips, config["demand"]["modes"], config["run"], config.get("random_seed")
    )
    truck_agents, truck_alts, truck_trips = generate_truck_agents(
        truck_trips, config["run"]["road_only"]["directory"], edges
    )
    agents, alts, trips, used_nodes = merge_populations(
        agents, alts, trips, truck_agents, truck_alts, truck_trips
    )

    metro_run.write_agents(run_directory, agents, alts, trips)

    all_nodes = set(edges["source"]).union(set(edges["target"]))
    if any(n not in all_nodes for n in used_nodes):
        print(
            "Warning: the origin / destination node of some trips is not a valid node of the road network"
        )

    metro_run.write_road_network(run_directory, edges, vehicles)

    write_parameters(
        run_directory,
        config["run"],
        config["run"]["road_only"]["directory"],
    )
