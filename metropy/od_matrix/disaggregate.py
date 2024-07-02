import os
import time
from collections import defaultdict

import numpy as np
import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io


def read_edges(filename: str, crs: str, config: dict):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(
        filename,
        columns=[
            "edge_id",
            "source",
            "target",
            "length",
            "road_type",
            "urban",
            "geometry",
        ],
    )
    gdf.to_crs(crs, inplace=True)
    if "weights" in config:

        def get_weight(row, weights):
            if row["urban"]:
                return weights["urban"][row["road_type"]]
            else:
                return weights["rural"][row["road_type"]]

        weights = config["weights"]
        gdf["weight"] = gdf[["road_type", "urban"]].apply(lambda r: get_weight(r, weights), axis=1)
    else:
        gdf["weight"] = 1
    # Get longitude, latitude of nodes.
    nodes = (
        gdf.drop_duplicates(subset=["source"]).set_index("source").to_crs("EPSG:4326")[["geometry"]]
    )
    nodes["lng"] = nodes.geometry.apply(lambda g: g.coords[0][0])
    nodes["lat"] = nodes.geometry.apply(lambda g: g.coords[0][1])
    df = pl.from_pandas(
        nodes[["lng", "lat"]],
        include_index=True,
        schema_overrides={"source": pl.UInt64},
    ).rename({"source": "node_id"})
    return gdf, df


def read_zones(filename: str, crs: str):
    print("Reading zones...")
    gdf = metro_io.read_geodataframe(filename, columns=["zone_id", "geometry"])
    gdf.to_crs(crs, inplace=True)
    return gdf


def match_edges_with_zones(edges: gpd.GeoDataFrame, zones: gpd.GeoDataFrame):
    print("Matching edges with zones...")
    n0 = len(edges)
    edges = edges.sjoin(zones, predicate="intersects", how="inner")
    n1 = edges["edge_id"].nunique()
    if n1 < n0:
        print(f"{n0 - n1:,} edges are not part of any zone")
    print("Computing length of edges in the matched zones...")
    edges["matched_length"] = edges.geometry.intersection(
        zones.set_index("zone_id").loc[edges["zone_id"]], align=False
    ).length
    df = pl.from_pandas(
        edges.loc[:, ["edge_id", "source", "target", "weight", "zone_id", "matched_length"]],
        schema_overrides={
            "edge_id": pl.UInt64,
            "source": pl.UInt64,
            "target": pl.UInt64,
            "weight": pl.Float64,
            "zone_id": pl.UInt64,
            "matched_length": pl.Float64,
        },
    )
    return df


def read_od_matrix(filename: str):
    print("Reading origin-destination matrix...")
    df = metro_io.read_dataframe(filename, columns=["origin", "destination", "count"])
    return df


def generate_origin_destination(
    df: pl.DataFrame, od_matrix: pl.DataFrame, config: dict, random_seed=None
):
    rng = np.random.default_rng(random_seed)
    if "max_nb_nodes_per_zone" in config:
        m: int = config["max_nb_nodes_per_zone"]
    else:
        m = len(df) + 1
    # Compute edge probability by zone.
    df = df.with_columns((pl.col("weight") * pl.col("matched_length")).alias("tot_weight"))
    # Use uniform probabilities if all weights are zero in a zone.
    df = df.with_columns(
        pl.when(pl.col("tot_weight").sum().over("zone_id") > 0.0)
        .then(pl.col("tot_weight"))
        .otherwise(pl.lit(1.0))
        .alias("prob")
    )
    matched_zones = set(df["zone_id"])
    zone_edges = (
        df.select("zone_id", "source", "target", "prob")
        .filter(pl.col("prob") > 0.0)
        .partition_by(["zone_id"], as_dict=True)
    )
    # Set `count` variable to integer.
    if od_matrix["count"].dtype.is_float():
        reminders = od_matrix["count"] % 1.0
        draws = pl.Series(rng.random(size=len(reminders)))
        values = (draws < reminders).cast(pl.UInt64)
        od_matrix = od_matrix.with_columns(pl.col("count").cast(pl.UInt64) + values)
    else:
        assert od_matrix["count"].dtype.is_integer()
    od_matrix = od_matrix.filter(pl.col("count") > 0)
    # Filter out zones without edge.
    all_zones = set(od_matrix["origin"]).union(set(od_matrix["destination"]))
    n0 = od_matrix["count"].sum()
    od_matrix = od_matrix.filter(
        pl.col("origin").is_in(matched_zones) & pl.col("destination").is_in(matched_zones)
    )
    n1 = od_matrix["count"].sum()
    if n1 < n0:
        print(f"Warning: {n0 - n1:,} trips cannot be generated as there is no edge in the zone")
        print("They represent {:.2%} of all trips".format((n0 - n1) / n0))
        missing_zones = all_zones.difference(matched_zones)
        print(f"Zones without edge:\n{missing_zones}")
    print("Drawing origins...")
    origin_counts = od_matrix.group_by("origin").agg(pl.col("count").sum())
    origins = dict()
    for zone_id, count in zip(origin_counts["origin"], origin_counts["count"]):
        zone_df = zone_edges[(zone_id,)]
        if len(zone_df) > m and count > m:
            # Randomly select `m` candidate nodes to be selected.
            idx = rng.choice(
                len(zone_df), p=zone_df["prob"] / zone_df["prob"].sum(), size=m, replace=False
            )
            zone_df = zone_df[idx]
        edges = rng.choice(len(zone_df), p=zone_df["prob"] / zone_df["prob"].sum(), size=count)
        # For first half, we use source, for the other half, we use target.
        n = count // 2
        origins[zone_id] = pl.concat((zone_df["source"][edges[:n]], zone_df["target"][edges[n:]]))
    print("Drawing destinations...")
    destination_counts = od_matrix.group_by("destination").agg(pl.col("count").sum())
    destinations = dict()
    for zone_id, count in zip(destination_counts["destination"], destination_counts["count"]):
        zone_df = zone_edges[(zone_id,)]
        if len(zone_df) > m and count > m:
            # Randomly select `m` candidate nodes to be selected.
            idx = rng.choice(
                len(zone_df), p=zone_df["prob"] / zone_df["prob"].sum(), size=m, replace=False
            )
            zone_df = zone_df[idx]
        edges = rng.choice(len(zone_df), p=zone_df["prob"] / zone_df["prob"].sum(), size=count)
        # For first half, we use source, for the other half, we use target.
        n = count // 2
        destinations[zone_id] = pl.concat(
            (zone_df["source"][edges[:n]], zone_df["target"][edges[n:]])
        )
    print("Generating trips...")
    origins_i = defaultdict(lambda: 0)
    destinations_j = defaultdict(lambda: 0)
    trips = list()
    for origin, destination, count in zip(
        od_matrix["origin"], od_matrix["destination"], od_matrix["count"]
    ):
        i = origins_i[origin]
        od_origins = origins[origin][i : i + count]
        origins_i[origin] += count
        j = destinations_j[destination]
        od_destinations = destinations[destination][j : j + count]
        destinations_j[destination] += count
        od_trips = list(zip(od_origins, od_destinations))
        trips.extend(od_trips)
    print(f"{len(trips):,} trips generated")
    trip_df = pl.DataFrame(
        trips, schema=[("origin_node", pl.UInt64), ("destination_node", pl.UInt64)]
    )
    trip_df = trip_df.with_columns(
        pl.Series(np.repeat(od_matrix["origin"], od_matrix["count"])).alias("origin_zone")
    )
    trip_df = trip_df.with_columns(
        pl.Series(np.repeat(od_matrix["destination"], od_matrix["count"])).alias("destination_zone")
    )
    return trip_df


def save_population(trips: pl.DataFrame, nodes: pl.DataFrame):
    print("Saving population...")
    trips = trips.join(nodes, left_on="origin_node", right_on="node_id").rename(
        {"lng": "origin_lng", "lat": "origin_lat"}
    )
    trips = trips.join(nodes, left_on="destination_node", right_on="node_id").rename(
        {"lng": "destination_lng", "lat": "destination_lat"}
    )
    trips = trips.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt64).alias("trip_id"),
        pl.int_range(pl.len(), dtype=pl.UInt64).alias("person_id"),
    )
    # Create dummy persons and households.
    persons = trips.select("person_id", pl.col("person_id").alias("household_id"))
    households = persons.select("household_id", pl.lit(1).alias("number_of_persons"))
    metro_io.save_dataframe(
        households, os.path.join(config["population_directory"], "households.parquet")
    )
    metro_io.save_dataframe(
        persons, os.path.join(config["population_directory"], "persons.parquet")
    )
    metro_io.save_dataframe(trips, os.path.join(config["population_directory"], "trips.parquet"))
    return trips


def save_trip_coordinates(trips: pl.DataFrame, directory: str):
    print("Saving trip coordinates...")
    origin_coords_gdf = gpd.GeoDataFrame(
        data=trips["trip_id"].to_pandas(),
        geometry=gpd.points_from_xy(trips["origin_lng"], trips["origin_lat"], crs="EPSG:4326"),
    )
    origin_coords_gdf.to_parquet(os.path.join(directory, "trip_origins.parquet"))
    del origin_coords_gdf
    destination_coords_gdf = gpd.GeoDataFrame(
        data=trips["trip_id"].to_pandas(),
        geometry=gpd.points_from_xy(
            trips["destination_lng"], trips["destination_lat"], crs="EPSG:4326"
        ),
    )
    destination_coords_gdf.to_parquet(os.path.join(directory, "trip_destinations.parquet"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "crs",
        "clean_edges_file",
        "population_directory",
        "od_matrix.zone_filename",
        "od_matrix.od_matrix_filename",
    ]
    check_keys(config, mandatory_keys)
    output_dir = config["population_directory"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    this_config = config["od_matrix"]

    t0 = time.time()
    edges, nodes = read_edges(config["clean_edges_file"], config["crs"], this_config)
    zones = read_zones(config["od_matrix"]["zone_filename"], config["crs"])
    df = match_edges_with_zones(edges, zones)
    od_matrix = read_od_matrix(config["od_matrix"]["od_matrix_filename"])
    df = generate_origin_destination(df, od_matrix, this_config, config.get("random_seed"))
    trips = save_population(df, nodes)
    save_trip_coordinates(trips, output_dir)
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
