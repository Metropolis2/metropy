import os
import time
import json
import subprocess

import numpy as np
import polars as pl
from scipy.stats import cramervonmises_2samp
from skopt import gp_minimize
from skopt.space import Real
import gower

import metropy.utils.mpl as mpl
import metropy.run.base as metro_run
import metropy.utils.io as metro_io


def optimization(
    trips: pl.LazyFrame, run_directory: str, metropolis_exec: str, distrs: pl.DataFrame
):
    def objective_function(theta):
        parameters = theta_to_parameters(theta)
        results = get_departure_time_choice(
            trips, run_directory, metropolis_exec, parameters, random_seed=13081996
        )
        _, mean_dist = cramervonmises_distance(results, distrs)
        print(f"Distance: {mean_dist:.6f}")
        return mean_dist

    # Define the parameter space
    space = [
        Real(low=1e-2, high=10.0, name="mu"),
        Real(low=0.0, high=1.0, name="beta_work"),
        Real(low=0.0, high=1.0, name="beta_education"),
        Real(low=0.0, high=1.0, name="beta_shop"),
        Real(low=0.0, high=1.0, name="beta_leisure"),
        Real(low=0.0, high=1.0, name="beta_other"),
        Real(low=0.0, high=1.0, name="gamma_work"),
        Real(low=0.0, high=1.0, name="gamma_education"),
        Real(low=0.0, high=1.0, name="gamma_shop"),
        Real(low=0.0, high=1.0, name="gamma_leisure"),
        Real(low=0.0, high=1.0, name="gamma_other"),
    ]
    # Perform Bayesian Optimization
    result = gp_minimize(objective_function, space, n_calls=200, random_state=13081996, verbose=True)
    assert result is not None
    # Optimal parameters
    theta_star = result.x
    print(f"Optimal parameters: {theta_star}")
    # Minimum KL divergence
    min_kl_divergence = result.fun
    print(f"Minimum KL divergence: {min_kl_divergence}")
    return result


def theta_to_parameters(theta):
    return {
        "mu": theta[0],
        "beta": {
            "work": theta[1],
            "education": theta[2],
            "shop": theta[3],
            "leisure": theta[4],
            "other": theta[5],
        },
        "gamma": {
            "work": theta[6],
            "education": theta[7],
            "shop": theta[8],
            "leisure": theta[9],
            "other": theta[10],
        },
    }


def process_trips(trips: pl.LazyFrame, directory: str):
    # Remove truck trips (not needed here).
    trips = trips.filter(~pl.col("is_truck"))
    # Add origin / destination zone.
    zones = metro_io.scan_dataframe(os.path.join(directory, "trip_zones.parquet")).with_columns(
        # TODO. This is specific to IDF and should be improved.
        pl.col("departement_origin")
        .replace_strict(
            {"75": 1, "77": 3, "78": 3, "91": 3, "92": 2, "93": 2, "94": 2, "95": 3}, default=3
        )
        .alias("area_origin"),
        pl.col("departement_destination")
        .replace_strict(
            {"75": 1, "77": 3, "78": 3, "91": 3, "92": 2, "93": 2, "94": 2, "95": 3}, default=3
        )
        .alias("area_destination"),
    )
    trips = trips.join(zones, on="trip_id")
    # Add desired activity start time and duration.
    desired_times_filename = os.path.join(directory, "desired_times.parquet")
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
    trips = trips.filter(pl.col("activity_duration") >= 0.0)
    return trips


def read_departure_time_distributions(filename: str):
    return metro_io.read_dataframe(filename)


def read_departure_time_cluster_centers(filename: str):
    return metro_io.read_dataframe(filename)


def classify_trips(lf: pl.LazyFrame, cluster_centers: pl.DataFrame):
    print("Computing Gower distance between trips and cluster centers...")
    variables = [
        "departure_time",
        "preceding_purpose",
        "following_purpose",
        "od_distance",
        "area_origin",
        "area_destination",
        "departement_origin",
        "departement_destination",
    ]
    x = lf.select(variables).collect().to_pandas()
    y = cluster_centers.select(variables).to_pandas()
    dists = gower.gower_matrix(
        x,
        y,
        cat_features=[False, True, True, False, True, True, True, True],
    )
    categories = dists.argmin(axis=1)
    lf = lf.with_columns(category=pl.Series(categories))
    return lf


def cramervonmises_distance(metro_trips: pl.DataFrame, distrs: pl.DataFrame):
    # Compute the Cramer-von Mises statistics for all categories.
    dists = dict()
    lengths = dict()
    distrs_dict = distrs.partition_by("category", as_dict=True, include_key=False)
    for (cat,), metro_df in metro_trips.partition_by(
        "category", as_dict=True, include_key=False
    ).items():
        this_distr = distrs_dict[(cat,)]
        values = np.repeat(
            this_distr["departure_time"].to_numpy(), this_distr["weight"].cast(pl.Int64).to_numpy()
        )
        res = cramervonmises_2samp(metro_df["departure_time"].to_numpy(), values)
        dists[cat] = res.statistic
        lengths[cat] = len(metro_df)
    # Compute the average Cramer-von Mises statistic, weighted by sample size.
    mean_dist = sum(dists[cat] * lengths[cat] for cat in dists.keys()) / sum(lengths.values())
    return dists, mean_dist


def initialize_simulation(run_directory: str, config: dict, congestion_run: str):
    edges = metro_run.read_edges(
        config["clean_edges_file"],
        config.get("routing", dict).get("road_split", dict).get("main_edges_filename"),
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
        config.get("capacities_filename"),
    )
    edges = metro_run.generate_edges(edges, config["run"])
    vehicles = metro_run.generate_vehicles(config["run"])
    metro_run.write_road_network(run_directory, edges, vehicles)
    write_parameters(run_directory, config["run"], congestion_run)


def write_parameters(run_directory: str, config: dict, congestion_run: str):
    parameters = metro_run.PARAMETERS.copy()
    parameters["learning_model"]["value"] = 0.0
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
    parameters["only_compute_decisions"] = True
    print("Writing parameters")
    with open(os.path.join(run_directory, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters))


def get_departure_time_choice(
    trips: pl.LazyFrame,
    run_directory: str,
    metropolis_exec: str,
    parameters: dict,
    random_seed=None,
):
    # Generate population.
    agents, alts, trips_df = generate_agents(trips, parameters, random_seed)
    # Save population.
    metro_run.write_agents(run_directory, agents, alts, trips_df)
    # Run simulator.
    print("Running simulation...")
    parameters_filename = os.path.join(run_directory, "parameters.json")
    subprocess.run(" ".join([metropolis_exec, parameters_filename]), shell=True)
    # Load results.
    tds = pl.scan_parquet(os.path.join(run_directory, "output", "trip_results.parquet"))
    results = trips.join(tds, on="trip_id").select(
        "category", pl.col("pre_exp_departure_time").alias("departure_time")
    )
    return results.collect()


def generate_agents(trips: pl.LazyFrame, parameters: dict, random_seed=None):
    print("Creating agent-level DataFrame")
    columns = trips.collect_schema().names()
    if "tour_id" in columns:
        id_col = "tour_id"
    else:
        assert "agent_id" in columns
        id_col = "agent_id"
    # Agent-level values: only the identifier is useful.
    agents = trips.select(pl.col(id_col).alias("agent_id")).unique().collect()

    print("Creating alternative-level DataFrame")
    # Alternative-level values.
    alts = (
        trips.group_by(pl.col(id_col).alias("agent_id"))
        .agg(
            pl.lit(1).alias("alt_id"),
            pl.col("access_time").first().alias("origin_delay"),
            pl.lit("Continuous").alias("dt_choice.type"),
            pl.lit("Logit").alias("dt_choice.model.type"),
            pl.lit(parameters["mu"]).alias("dt_choice.model.mu"),
            pl.lit(1.0).alias("total_travel_utility.one"),
            pl.lit(False).alias("pre_compute_route"),
        )
        .collect()
    )
    # Add random uniforms for departure-time choice.
    rng = np.random.default_rng(random_seed)
    uu = rng.random(size=len(alts))
    alts = alts.with_columns(pl.Series(uu).alias("dt_choice.model.u"))

    print("Creating trip-level DataFrame")
    # Trip-level values.
    trips_df = trips.select(
        pl.col(id_col).alias("agent_id"),
        pl.lit(1).alias("alt_id"),
        pl.col("trip_id"),
        # Stopping time is activity duration + egress time + access time of next trip.
        (
            pl.col("activity_duration")
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
        .otherwise(None)
        .alias("class.travel_time"),
        pl.when(pl.col("secondary_only"))
        .then(None)
        .otherwise(pl.col("access_node"))
        .alias("class.origin"),
        pl.when(pl.col("secondary_only"))
        .then(None)
        .otherwise(pl.col("egress_node"))
        .alias("class.destination"),
        pl.when(pl.col("secondary_only")).then(None).otherwise(1).alias("class.vehicle"),
        pl.lit("AlphaBetaGamma").alias("schedule_utility.type"),
        # The desired arrival time at trips' destination is the desired activity start time minus
        # the egress time.
        (pl.col("activity_start_time") - pl.col("egress_time").fill_null(0.0)).alias(
            "schedule_utility.tstar"
        ),
        pl.col("following_purpose")
        .replace_strict(parameters["beta"], default=0.0)
        .alias("schedule_utility.beta"),
        pl.col("following_purpose")
        .replace_strict(parameters["gamma"], default=0.0)
        .alias("schedule_utility.gamma"),
    ).collect()
    assert trips_df.select(
        (pl.col("class.origin").is_not_null() | pl.col("class.travel_time").is_not_null()).all()
    ).item(), "The origin / destination is unknown for some trips"
    assert trips_df["schedule_utility.tstar"].is_null().sum() == 0

    print(
        "Generated {:,} agents, with {:,} alternatives and {:,} trips ({:,} being road trips)".format(
            agents.height,
            alts.height,
            trips_df.height,
            trips_df.select(pl.col("class.type") == "Road").sum().item(),
        )
    )
    return agents, alts, trips_df


def make_graphs(
    metro_trips: pl.DataFrame, distrs: pl.DataFrame, graph_dir: str, period: list[float]
):
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    # Global departure-time distribution.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = np.arange(period[0] / 3600, (period[1] + 1.0) / 3600, 15 / 60)
    ax.hist(
        distrs["departure_time"] / 3600,
        bins=list(bins),
        weights=distrs["weight"],
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(0),
        label="Travel Survey",
    )
    ax.hist(
        metro_trips["departure_time"] / 3600,
        bins=list(bins),
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.7,
        color=mpl.CMP(1),
        label="METROPOLIS",
    )
    ax.set_xlim(period[0] / 3600, period[1] / 3600)
    ax.set_xlabel("Departure time (h)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative density")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "all_departure_times_hist.pdf"))
    # Departure-time distribution by category.
    distrs_dict = distrs.partition_by("category", as_dict=True, include_key=False)
    for (cat,), metro_df in metro_trips.partition_by(
        "category", as_dict=True, include_key=False
    ).items():
        this_distr = distrs_dict[(cat,)]
        fig, ax = mpl.get_figure(fraction=0.8)
        bins = np.arange(period[0] / 3600, (period[1] + 1.0) / 3600, 15 / 60)
        ax.hist(
            this_distr["departure_time"] / 3600,
            bins=list(bins),
            weights=this_distr["weight"],
            density=True,
            cumulative=True,
            histtype="step",
            alpha=0.7,
            color=mpl.CMP(0),
            label="Travel Survey",
        )
        ax.hist(
            metro_df["departure_time"] / 3600,
            bins=list(bins),
            density=True,
            cumulative=True,
            histtype="step",
            alpha=0.7,
            color=mpl.CMP(1),
            label="METROPOLIS",
        )
        ax.set_xlim(period[0] / 3600, period[1] / 3600)
        ax.set_xlabel("Departure time (h)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cumulative density")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(graph_dir, f"departure_times_hist_{cat}.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "population_directory",
        "metropolis_exec",
        "run.period",
        "routing.road_split.main_edges_filename",
        "routing.road_split.trips_filename",
        "calibration.free_flow_calibration.output_filename",
        "calibration.departure_time_calibration.run_directory",
        "travel_survey.departure_time_distribution.output_filename",
        "graph_directory",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    run_directory = config["calibration"]["departure_time_calibration"]["run_directory"]
    if not os.path.isdir(run_directory):
        os.makedirs(os.path.join(run_directory, "input"))
    initialize_simulation(
        run_directory,
        config,
        config["run"]["road_only"]["directory"],
    )

    trips = metro_run.read_trips(
        config["population_directory"],
        config["routing"]["road_split"]["trips_filename"],
        config["run"]["period"],
        road_only=True,
    )

    trips = process_trips(trips, config["population_directory"])

    distrs = read_departure_time_distributions(
        config["travel_survey"]["departure_time_distribution"]["distribution_filename"]
    )
    centers = read_departure_time_cluster_centers(
        config["travel_survey"]["departure_time_distribution"]["cluster_filename"]
    )

    trips = classify_trips(trips, centers)

    _, mean_dist = cramervonmises_distance(
        trips.select("category", "departure_time").collect(), distrs
    )
    print(
        f"Cramer-von Mises distance between synthetic population and travel survey: {mean_dist:.6f}"
    )

    res = optimization(trips, run_directory, config["metropolis_exec"], distrs)

    # Re-run the optimal to get departure times.
    metro_trips = get_departure_time_choice(
        trips,
        run_directory,
        config["metropolis_exec"],
        parameters=theta_to_parameters(res.x),
        random_seed=13081996,
    )

    graph_dir = os.path.join(config["graph_directory"], "calibration.departure_time_calibration")
    make_graphs(metro_trips, distrs, graph_dir, config["run"]["period"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
