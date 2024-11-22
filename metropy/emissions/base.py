import os

import numpy as np
import polars as pl
import geopandas as gpd
from shapely.geometry import Polygon

import metropy.utils.io as metro_io


def read_edges(filename: str, edge_penalties_filename=None):
    print("Reading edges...")
    lf = metro_io.scan_dataframe(filename)
    if edge_penalties_filename is not None:
        edges_penalties = metro_io.scan_dataframe(edge_penalties_filename)
        columns = edges_penalties.collect_schema().names()
        assert (
            "additive_penalty" in columns
        ), "No column `additive_penalty` in the edges penalties file"
        if "speed" in columns:
            lf = lf.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("additive_penalty").fill_null(pl.lit(0.0)),
                pl.col("speed").fill_null(pl.col("speed_limit")),
            )
        else:
            print("Warning: no `speed` column in the edges penalties, using additive penalty only")
            lf = lf.join(edges_penalties, on="edge_id", how="left").with_columns(
                pl.col("additive_penalty").fill_null(pl.lit(0.0)),
                pl.col("speed_limit").alias("speed"),
            )
    else:
        lf = lf.with_columns(additive_penalty=pl.lit(0.0), speed=pl.col("speed_limit"))
    # Compute free-flow travel time (in seconds).
    lf = lf.with_columns(
        ff_travel_time=pl.col("additive_penalty") + pl.col("length") / (pl.col("speed") / 3.6)
    )
    # Set length in km.
    lf = lf.with_columns(pl.col("length") / 1000)
    df = lf.select("edge_id", "length", "ff_travel_time").collect()
    return df


def read_routes(
    run_directory: str,
    road_split_filename: str | None,
    edges: pl.DataFrame,
    modes: list[str],
    interval: float,
    include_car_passenger=False,
):
    columns = ("tour_id", "trip_id", "edge_id", "entry_time", "exit_time", "travel_time", "length")
    print("Reading routes...")
    mode_ids = [modes.index("car_driver")]
    if include_car_passenger:
        mode_ids.append(modes.index("car_passenger"))
    # Find tours with car driver as mode of transportation.
    car_driver_tours = (
        pl.scan_parquet(os.path.join(run_directory, "output", "agent_results.parquet"))
        .filter(pl.col("agent_id") < 100_000_000, pl.col("selected_alt_id").is_in(mode_ids))
        .select(tour_id=pl.col("agent_id"))
        .collect()
        .to_series()
    )
    # Find the trip departure times (required for secondary road trips).
    df = (
        pl.scan_parquet(os.path.join(run_directory, "output", "trip_results.parquet"))
        .rename({"agent_id": "tour_id"})
        .filter(pl.col("tour_id").is_in(car_driver_tours))
        .select("tour_id", "trip_id", "departure_time", secondary_only=pl.col("nb_edges").is_null())
        .collect()
    )
    # Routes for trips on main network (excluding access / egress part).
    main_routes = (
        pl.scan_parquet(os.path.join(run_directory, "output", "route_results.parquet"))
        .rename({"agent_id": "tour_id"})
        .filter(pl.col("tour_id").is_in(car_driver_tours))
        .sort("tour_id", "trip_id", "entry_time")
        .with_columns(travel_time=pl.col("exit_time") - pl.col("entry_time"))
        .join(edges.lazy(), on="edge_id")
        .select(columns)
        .collect()
    )
    if road_split_filename is not None:
        road_split = metro_io.scan_dataframe(road_split_filename)
        # Routes for secondary trips.
        secondary_routes = (
            df.lazy()
            .filter("secondary_only")
            .join(
                road_split.select("trip_id", "route"),
                on="trip_id",
                how="inner",
            )
            .explode("route")
            .rename({"route": "edge_id"})
            .join(edges.lazy(), on="edge_id")
            .rename({"ff_travel_time": "travel_time"})
            .with_columns(
                exit_time=pl.col("departure_time")
                + pl.col("travel_time").cum_sum().over("tour_id", "trip_id")
            )
            .with_columns(entry_time=pl.col("exit_time") - pl.col("travel_time"))
            .select(columns)
            .collect()
        )
        # Access parts of main-network trips.
        access_parts = (
            main_routes.group_by("tour_id", "trip_id")
            .agg(departure_time=pl.col("entry_time").first())
            .lazy()
            .join(
                road_split.select("trip_id", "access_part"),
                on="trip_id",
                how="inner",
            )
            .explode("access_part")
            .rename({"access_part": "edge_id"})
            .join(edges.lazy(), on="edge_id")
            .rename({"ff_travel_time": "travel_time"})
            .with_columns(
                entry_time=pl.col("departure_time")
                - pl.col("travel_time").cum_sum(reverse=True).over("tour_id", "trip_id")
            )
            .with_columns(exit_time=pl.col("entry_time") + pl.col("travel_time"))
            .select(columns)
            .collect()
        )
        # Egress parts of main-network trips.
        egress_parts = (
            main_routes.group_by("tour_id", "trip_id")
            .agg(arrival_time=pl.col("exit_time").last())
            .lazy()
            .join(
                road_split.select("trip_id", "egress_part"),
                on="trip_id",
                how="inner",
            )
            .explode("egress_part")
            .rename({"egress_part": "edge_id"})
            .join(edges.lazy(), on="edge_id")
            .rename({"ff_travel_time": "travel_time"})
            .with_columns(
                exit_time=pl.col("arrival_time")
                + pl.col("travel_time").cum_sum().over("tour_id", "trip_id")
            )
            .with_columns(entry_time=pl.col("exit_time") - pl.col("travel_time"))
            .select(columns)
            .collect()
        )
        routes = pl.concat(
            (secondary_routes, access_parts, main_routes, egress_parts), how="vertical"
        ).sort("tour_id", "trip_id", "entry_time")
    else:
        if df["secondary_only"].not_().any():
            print(
                "Warning: No road split file was provided but there are some secondary-only trips"
            )
        routes = main_routes
    # Add time period.
    routes = routes.with_columns(time_period=(pl.col("exit_time") // interval).cast(pl.UInt8))
    return routes


def read_edge_length(filename: str, crs: str):
    print("Reading edge lengths...")
    edges = metro_io.read_geodataframe(filename, columns=["edge_id", "geometry"])
    edges.to_crs(crs, inplace=True)
    # We re-compute edges' length even though it is already in the data because the length might be
    # lowe-bounded by some value (1m by default).
    edges["length"] = edges["geometry"].length
    return edges


def create_grid(edges: gpd.GeoDataFrame, resolution: float, crs: str):
    print("Creating grid...")
    half_resolution = resolution / 2
    minx, miny, maxx, maxy = edges.total_bounds
    minx = resolution * np.floor(minx / resolution)
    miny = resolution * np.floor(miny / resolution)
    maxx = resolution * np.ceil(maxx / resolution)
    maxy = resolution * np.ceil(maxy / resolution)
    xs = np.arange(minx + half_resolution, maxx, resolution)
    ys = np.arange(miny + half_resolution, maxy, resolution)
    polygons = []
    for x in xs:
        for y in ys:
            polygons.append(
                Polygon(
                    [
                        (x - half_resolution, y - half_resolution),
                        (x + half_resolution, y - half_resolution),
                        (x + half_resolution, y + half_resolution),
                        (x - half_resolution, y + half_resolution),
                    ]
                )
            )
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    grid_gdf["cell_id"] = np.arange(len(grid_gdf))
    grid = pl.DataFrame(
        {
            "cell_id": grid_gdf["cell_id"],
            "x0": grid_gdf.geometry.centroid.x,
            "y0": grid_gdf.geometry.centroid.y,
        }
    )
    print(f"Created a grid with {len(grid):,} polygons")
    return grid_gdf, grid


def get_gridded_edges(edges: gpd.GeoDataFrame, grid: gpd.GeoDataFrame):
    print("Computing edges / cells intersections...")
    gdf = gpd.sjoin(edges, grid, predicate="intersects")
    # Retrieve geometries of the matched cells.
    cell_geoms = grid.loc[gdf["index_right"], "geometry"]
    # Compute length of the intersection between the edges and the cells (in meters).
    gdf["match_length"] = gdf["geometry"].intersection(cell_geoms, align=False).length
    # Compute share of length on the cell, for each edge.
    gdf["length_share"] = gdf["match_length"] / gdf["length"]
    df = pl.from_pandas(gdf[["edge_id", "length_share", "cell_id"]])
    df = df.with_columns(
        x0=pl.Series(cell_geoms.centroid.x),
        y0=pl.Series(cell_geoms.centroid.y),
    )
    nb_cells = df["cell_id"].n_unique()
    print(f"Number of cells with at least one edge: {nb_cells:,}")
    return df
