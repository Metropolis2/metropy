import os
import time

import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io
import metropy.emissions.base as metro_emissions


def compute_exposure_locations(
    population_directory: str,
    run_directory: str,
    road_split_filename: str,
    grid: pl.DataFrame,
    gridded_edges: pl.DataFrame,
    routes: pl.DataFrame,
    period: list[float],
    resolution: float,
    crs: str,
):
    # TODO: In the current version, `time_period` is not used.
    persons = pl.scan_parquet(os.path.join(population_directory, "persons.parquet"))
    trips = pl.scan_parquet(os.path.join(population_directory, "trips.parquet"))
    person_tour_map = trips.select("person_id", "tour_id").unique().collect()
    enum = pl.Enum(["home", "en_trip", "activity"])
    period_length = period[1] - period[0]

    print("Reading activities locations...")
    trips = pl.read_parquet(
        os.path.join(population_directory, "trips.parquet"),
        columns=[
            "person_id",
            "trip_id",
            "origin_lng",
            "origin_lat",
            "destination_lng",
            "destination_lat",
        ],
    )
    origins = gpd.GeoSeries.from_xy(trips["origin_lng"], trips["origin_lat"], crs="EPSG:4326")
    origins = origins.to_crs(crs)
    trips = trips.select(
        "person_id",
        "trip_id",
        x_origin=pl.Series(origins.x),
        y_origin=pl.Series(origins.y),
    )

    print("Computing en-route exposure locations...")
    # Compute, for each tour, how much time (in seconds) is spent in each cell (as part of the road
    # trip).
    en_trip = (
        routes.lazy()
        .join(gridded_edges.lazy(), on="edge_id")
        .join(person_tour_map.lazy(), on="tour_id")
        .group_by("person_id", "cell_id")
        .agg(time_spent=(pl.col("length_share") * pl.col("travel_time")).sum())
        .select(
            "person_id",
            "cell_id",
            share=pl.col("time_spent") / period_length,
            category=pl.lit("en_trip", dtype=enum),
        )
        .collect()
    )

    print("Computing activity exposure locations...")
    road_tour_ids = routes["tour_id"].unique()
    trip_timings = (
        pl.scan_parquet(os.path.join(run_directory, "output", "trip_results.parquet"))
        .rename({"agent_id": "tour_id"})
        .join(trips.lazy(), on="trip_id")
        .with_columns(
            road_trip=pl.col("tour_id").is_in(road_tour_ids),
            middle_time=(pl.col("departure_time") + pl.col("arrival_time")) / 2,
        )
        .join(pl.scan_parquet(road_split_filename), on="trip_id", how="left")
        # For non-road trips, the agents are teleported from origin to destination at the middle of
        # the trip.
        .with_columns(
            departure_time=pl.when("road_trip")
            .then(pl.col("departure_time") - pl.col("access_time").fill_null(0.0))
            .otherwise("middle_time"),
            arrival_time=pl.when("road_trip")
            .then(pl.col("arrival_time") + pl.col("egress_time").fill_null(0.0))
            .otherwise("middle_time"),
        )
        .select(
            "person_id",
            "tour_id",
            "departure_time",
            "arrival_time",
            "x_origin",
            "y_origin",
        )
        .sort("person_id", "departure_time")
        .collect()
    )
    # Find the start / end time of all activities in the middle of a tour.
    # `drop_nulls` will drop the first activity because of `shift(1)` creating nulls.
    activities = (
        trip_timings.lazy()
        .select(
            "person_id",
            start_time=pl.col("arrival_time").shift(1).over("tour_id"),
            end_time=pl.col("departure_time"),
            x=pl.col("x_origin"),
            y=pl.col("y_origin"),
        )
        .drop_nulls()
        .with_columns(
            duration=pl.col("end_time") - pl.col("start_time"),
            x0=resolution * pl.col("x").floordiv(resolution) + resolution / 2.0,
            y0=resolution * pl.col("y").floordiv(resolution) + resolution / 2.0,
        )
        .join(grid.lazy(), on=["x0", "y0"])
        .select(
            "person_id",
            "cell_id",
            share=pl.col("duration") / period_length,
            category=pl.lit("activity", dtype=enum),
        )
        .collect()
    )

    print("Computing home exposure locations...")
    # Total time spent not at home for all agents.
    not_home_times = (
        trip_timings.lazy()
        .group_by("tour_id")
        .agg(
            pl.col("person_id").first(),
            tour_start=pl.col("departure_time").first(),
            tour_end=pl.col("arrival_time").last(),
        )
        .with_columns(
            tour_time=pl.col("tour_end") - pl.col("tour_start"),
        )
        .group_by("person_id")
        .agg(not_home_time=pl.col("tour_time").sum())
        .collect()
    )
    homes = (
        persons.join(not_home_times.lazy(), on="person_id", how="left")
        .with_columns(pl.col("not_home_time").fill_null(0.0))
        .join(
            pl.scan_parquet(os.path.join(population_directory, "households.parquet")),
            on="household_id",
            how="inner",
        )
        .select("person_id", "not_home_time", "geometry")
        .collect()
    )
    home_geoms = gpd.GeoSeries.from_wkb(homes["geometry"])
    # Identify the cell in which their home is located.
    homes = (
        homes.lazy()
        .with_columns(
            x=pl.Series(home_geoms.x),
            y=pl.Series(home_geoms.y),
        )
        .with_columns(
            x0=resolution * pl.col("x").floordiv(resolution) + resolution / 2.0,
            y0=resolution * pl.col("y").floordiv(resolution) + resolution / 2.0,
        )
        .join(grid.lazy(), on=["x0", "y0"])
        .select(
            "person_id",
            "cell_id",
            # Note. The share can actually be negative if the total time on tours is larger than the
            # period time.
            # In this case, the exposure at home will reduce the exposure on tour.
            share=(period_length - pl.col("not_home_time")) / period_length,
            category=pl.lit("home", dtype=enum),
        )
        .collect()
    )

    df = pl.concat((en_trip, activities, homes), how="vertical")
    df = df.with_columns(
        duration_h=period_length * pl.col("share") / 3600, time_period=pl.lit(0, dtype=pl.UInt8)
    )
    return df


def read_concentrations(filename: str, pollutants: list[str]):
    print("Reading concentrations...")
    df = pl.read_parquet(
        filename, columns=["cell_id", "time_period", *pollutants], use_pyarrow=True
    )
    return df


def compute_exposure(
    df: pl.DataFrame,
    df_concentration: pl.DataFrame,
    pollutants: list[str],
    config: dict,
    simulation_ratio: float,
):
    df = (
        df.lazy()
        .join(df_concentration.lazy(), on=["cell_id", "time_period"])
        .group_by("person_id", "cell_id", "time_period", "category")
        .agg(
            [
                (pl.col(p) * pl.col("duration_h") * config["exposure_probs"][p]).sum().alias(p)
                for p in pollutants
            ]
        )
        .collect()
    )
    print("Total number of premature deaths:\n")
    print(df.select(pl.col(p).sum() / simulation_ratio for p in pollutants))
    df = df.with_columns(
        pl.col(p)
        * config["average_years_lost"]
        * config["year_of_life_lost_price"]
        for p in pollutants
    )
    print("Total exposure cost by pollutant:")
    print(df.select(pl.col(p).sum() / simulation_ratio for p in pollutants))
    tot = df.select(pl.sum_horizontal(pl.col(p).sum() / simulation_ratio for p in pollutants)).item()
    print(f"Total exposure cost (all pollutants): {tot:,.0f} €")
    print("Exposure cost by category:")
    print(
        df.group_by("category").agg(
            *(pl.col(p).sum() / simulation_ratio for p in pollutants),
            total=sum(pl.col(p).sum() / simulation_ratio for p in pollutants),
        )
    )
    print("Exposure cost by time period:")
    print(
        df.group_by("time_period").agg(
            *(pl.col(p).sum() / simulation_ratio for p in pollutants),
            total=sum(pl.col(p).sum() / simulation_ratio for p in pollutants),
        )
    )
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "population_directory",
        "run_directory",
        "crs",
        "run.period",
        "emisens.interval",
        "metro-trace.grid_resolution",
        "metro-trace.concentration_filename",
        "metro-trace.population_dispersion_filename",
        "metro-trace.exposure_filename",
        "metro-trace.exposure_probs",
        "metro-trace.average_years_lost",
        "metro-trace.year_of_life_lost_price",
        "routing.road_split.trips_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    edge_length = metro_emissions.read_edge_length(config["clean_edges_file"], config["crs"])

    grid_gdf, grid = metro_emissions.create_grid(
        edge_length, config["metro-trace"]["grid_resolution"], config["crs"]
    )

    gridded_edges = metro_emissions.get_gridded_edges(edge_length, grid_gdf)

    edges = metro_emissions.read_edges(
        config["clean_edges_file"],
        config.get("calibration", dict).get("free_flow_calibration", dict).get("output_filename"),
    )

    routes = metro_emissions.read_routes(
        config["run_directory"],
        config.get("routing", dict).get("road_split", dict).get("trips_filename"),
        edges,
        config["demand"]["modes"],
        config["emisens"]["interval"],
        include_car_passenger=True,
    )

    df = compute_exposure_locations(
        config["population_directory"],
        config["run_directory"],
        config["routing"]["road_split"]["trips_filename"],
        grid,
        gridded_edges,
        routes,
        config["run"]["period"],
        config["metro-trace"]["grid_resolution"],
        config["crs"],
    )

    metro_io.save_dataframe(df, config["metro-trace"]["population_dispersion_filename"])

    pollutants = list(config["metro-trace"]["exposure_probs"].keys())

    df_concentration = read_concentrations(
        config["metro-trace"]["concentration_filename"], pollutants
    )

    df = compute_exposure(
        df,
        df_concentration,
        pollutants,
        config["metro-trace"],
        config.get("run", dict).get("simulation_ratio", 1.0),
    )

    metro_io.save_dataframe(df, config["metro-trace"]["exposure_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
