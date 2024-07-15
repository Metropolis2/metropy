import os
from collections import defaultdict
from zipfile import ZipFile
from itertools import pairwise

import numpy as np
import polars as pl
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import metropy.utils.io as metro_io


def read_pt_itineraries(filename: str, route_id: str):
    print("Reading public-transit itineraries...")
    df = metro_io.scan_dataframe(filename)
    # Remove the 2 first characters of the route_id and stop_id, as they are added by OTP.
    df = df.with_columns(
        pl.col("legs").list.eval(
            pl.struct(
                pl.element().struct.field("leg_index"),
                pl.element().struct.field("mode"),
                pl.element().struct.field("travel_time"),
                pl.element().struct.field("route_id").str.slice(2),
                pl.element().struct.field("from_stop_id").str.slice(2),
                pl.element().struct.field("to_stop_id").str.slice(2),
            )
        )
    )
    df = df.filter(
        pl.col("legs").list.eval(pl.element().struct.field("route_id") == route_id).list.any()
    )
    df = df.with_columns(
        pl.col("legs").list.len().alias("nb_legs"),
        (
            pl.col("travel_time")
            - pl.col("legs").list.eval(pl.element().struct.field("travel_time")).list.sum()
        ).alias("waiting_time"),
    )
    return df


def filter_pt_trips(df: pl.LazyFrame, run_directory: str, pt_alt_id: int):
    agent_results = metro_io.scan_dataframe(
        os.path.join(run_directory, "output", "agent_results.parquet")
    )
    trip_results = metro_io.scan_dataframe(
        os.path.join(run_directory, "output", "trip_results.parquet")
    )
    pt_agents = agent_results.filter(pl.col("selected_alt_id") == pt_alt_id)
    pt_trips = trip_results.join(pt_agents, on="agent_id", how="semi")
    df = df.join(pt_trips, on="trip_id", how="semi")
    return df


def find_file(zipfile, filename):
    for file in zipfile.filelist:
        if file.filename.endswith(filename):
            return zipfile.open(file.filename)


def read_as_dataframe(gtfs_zipfile: str, name: str):
    zipfile = ZipFile(gtfs_zipfile)
    file = find_file(zipfile, name)
    if file is None:
        raise Exception(f"Missing file: {name}")
    df = pl.read_csv(file.read())
    return df


def get_connections(df: pl.DataFrame, gtfs_zipfile: str):
    print("Finding all stop connections...")
    unique_route_ids = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("route_id"))
            .explode()
            .alias("route_id")
            .unique()
        )
        .collect()
        .get_column("route_id")
    )
    # Trips of the routes.
    trips = read_as_dataframe(gtfs_zipfile, "trips.txt").filter(
        pl.col("route_id").is_in(unique_route_ids)
    )
    # Stop times of the trips.
    stop_times = (
        read_as_dataframe(gtfs_zipfile, "stop_times.txt")
        .lazy()
        .join(trips.lazy(), on="trip_id", how="inner")
    )
    # Remove trips with duplicate sequence of stops (to do so, we compute the hash of the list of
    # hash stop id sequence).
    unique_trips = (
        stop_times.group_by("trip_id")
        .agg(
            pl.col("stop_id").sort_by("stop_sequence").alias("stops"),
            pl.col("stop_id").sort_by("stop_sequence").hash().alias("hashed_stop_sequence"),
            pl.col("route_id").first(),
        )
        .with_columns(pl.col("hashed_stop_sequence").hash())
        .unique(subset=["route_id", "hashed_stop_sequence"])
    )
    # Find all the direct connections and indirect connections for each trip.
    trip_connections = unique_trips.with_columns(
        pl.col("stops")
        .map_elements(get_direct_connections, return_dtype=pl.List(pl.String))
        .alias("direct_connections"),
        pl.col("stops")
        .map_elements(get_indirect_connections, return_dtype=pl.List(pl.String))
        .alias("indirect_connections"),
    )
    # Find all direct connections (excluding shortcuts) by route.
    # To do so, we compute the set difference between direct connections and indirect connections.
    connections = (
        trip_connections.group_by("route_id")
        .agg(pl.col("direct_connections").flatten(), pl.col("indirect_connections").flatten())
        .select(
            pl.col("route_id"),
            pl.col("direct_connections").list.set_difference(pl.col("indirect_connections")),
        )
        .explode("direct_connections")
        .with_columns(
            pl.col("direct_connections")
            .str.split_exact("->", 1)
            .struct.rename_fields(["from", "to"])
        )
        .unnest("direct_connections")
    )
    # Add stop name, latitude and longitude.
    columns = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    stops = read_as_dataframe(gtfs_zipfile, "stops.txt").select(columns)
    connections = connections.join(
        stops.lazy().rename({col: f"{col}_from" for col in columns}),
        left_on="from",
        right_on="stop_id_from",
        how="left",
        coalesce=True,
    ).join(
        stops.lazy().rename({col: f"{col}_to" for col in columns}),
        left_on="to",
        right_on="stop_id_to",
        how="left",
        coalesce=True,
    )
    # Add route name, route type and route color.
    routes = read_as_dataframe(gtfs_zipfile, "routes.txt").select(
        "route_id",
        "agency_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
    )
    connections = connections.join(
        routes.lazy(), on="route_id", how="left", coalesce=True
    ).with_columns(pl.col("route_color").str.replace("^", "#"))
    return connections.collect()


def filter_out(df: pl.DataFrame, connections: pl.DataFrame, config: dict):
    if "from_stop_id" in config and "to_stop_id" in config:
        print("Filtering valid trips...")
        pairs = (
            df.lazy()
            .explode("legs")
            .filter(pl.col("legs").struct.field("route_id").eq(config["route_id"]))
            .select(
                "trip_id",
                pl.col("legs").struct.field("from_stop_id"),
                pl.col("legs").struct.field("to_stop_id"),
            )
            .collect()
        )
        unique_pairs = pairs.unique(subset=["from_stop_id", "to_stop_id"])
        route_connects = connections.filter(pl.col("route_id") == config["route_id"]).select(
            pl.struct("from", "to").alias("pair")
        )
        graph = get_route_graph(route_connects["pair"])
        paths = dict(nx.all_pairs_dijkstra_path(graph))
        to_select = list()
        for from_id, to_id in zip(unique_pairs["from_stop_id"], unique_pairs["to_stop_id"]):
            path = paths[from_id][to_id]
            try:
                i = path.index(config["from_stop_id"])
                if config["to_stop_id"] in path[i + 1 :]:
                    to_select.append(True)
                else:
                    to_select.append(False)
            except ValueError:
                to_select.append(False)
        valid_pairs = unique_pairs.filter(pl.Series(to_select))
        trip_ids = (
            pairs.join(valid_pairs, on=["from_stop_id", "to_stop_id"], how="semi")
            .select("trip_id")
            .to_series()
        )
        df = df.filter(pl.col("trip_id").is_in(trip_ids))
    print("Number of valid trips: {:,}".format(len(df)))
    return df


def get_flows(df: pl.DataFrame, connections: pl.DataFrame):
    print("Computing connection-level flows...")
    # Count occurences of the unique tuples (route_id, from_stop, to_stop).
    columns = ["route_id", "from", "to"]
    raw_pairs = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("route_id"))
            .explode()
            .alias("route_id"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("from_stop_id"))
            .explode()
            .alias("from"),
            pl.col("legs").list.eval(pl.element().struct.field("to_stop_id")).explode().alias("to"),
        )
        .filter(pl.col("from").is_not_null())
        .group_by(columns)
        .agg(pl.len().alias("count"))
        .collect()
    )
    route_counts = raw_pairs.partition_by(["route_id"], as_dict=True)
    # Create a struct with all pairs (from, to).
    pairs = connections.select("route_id", pl.struct("from", "to").alias("pair")).filter(
        pl.col("route_id").is_in(raw_pairs["route_id"])
    )
    i = 0
    pair_counts = defaultdict(lambda: 0)
    for (route_id,), route_pairs in pairs.group_by(["route_id"]):
        this_route_counts = route_counts[(route_id,)]
        graph = get_route_graph(route_pairs["pair"])
        paths = dict(nx.all_pairs_dijkstra_path(graph))
        for _, from_stop, to_stop, count in this_route_counts.iter_rows():
            try:
                path = paths[from_stop][to_stop]
            except KeyError:
                i += count
            else:
                for from_stop, to_stop in pairwise(path):
                    pair_counts[(route_id, from_stop, to_stop)] += count
    if i:
        msg = "Warning. Cannot find path in route for {} segments (representing {:.2%} of segments)"
        print(msg.format(i, i / raw_pairs["count"].sum()))
    flows = pl.from_dicts(
        [
            {"route_id": route_id, "from": from_stop, "to": to_stop, "count": count}
            for (route_id, from_stop, to_stop), count in pair_counts.items()
        ]
    )
    return flows


def to_geopandas(connections):
    print("Building GeoDataFrame...")
    linestrings = (
        connections.select("stop_lon_from", "stop_lat_from", "stop_lon_to", "stop_lat_to")
        .to_struct("coords")
        .map_elements(get_linestring, return_dtype=pl.Object)
    )
    gdf = gpd.GeoDataFrame(connections.to_pandas(), geometry=linestrings.to_list(), crs="EPSG:4326")
    return gdf


def get_route_graph(route_pairs):
    G = nx.DiGraph()
    for pair in route_pairs:
        G.add_edge(pair["from"], pair["to"])
    return G


def plot_line(route_df):
    G = nx.DiGraph()
    coords = dict()
    names = dict()
    for row in route_df.iter_rows(named=True):
        G.add_edge(row["from"], row["to"])
        coords[row["from"]] = (row["stop_lon_from"], row["stop_lat_from"])
        coords[row["to"]] = (row["stop_lon_to"], row["stop_lat_to"])
        names[row["from"]] = row["stop_name_from"]
        names[row["to"]] = row["stop_name_to"]
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.set_axis_off()
    nx.draw_networkx(G, pos=coords, arrows=True, with_labels=True, labels=names, ax=ax)
    return fig


def get_direct_connections(stop_list):
    return [f"{from_stop}->{to_stop}" for from_stop, to_stop in pairwise(stop_list)]


def get_indirect_connections(stop_list):
    unique_stops = set(stop_list)
    if len(stop_list) == len(unique_stops):
        return [
            f"{from_stop}->{to_stop}"
            for i, from_stop in enumerate(stop_list[:-2])
            for to_stop in stop_list[i + 2 :]
        ]
    else:
        # There is a loop in the trip.
        unique, counts = np.unique(stop_list, return_counts=True)
        looped_stops = set(unique[counts > 1])
        assert len(looped_stops) > 0
        return [
            f"{from_stop}->{to_stop}"
            for i, from_stop in enumerate(stop_list[:-2])
            for to_stop in stop_list[i + 2 :]
            if from_stop not in looped_stops and to_stop not in looped_stops
        ]


def get_linestring(coords):
    return LineString(
        [
            [coords["stop_lon_from"], coords["stop_lat_from"]],
            [coords["stop_lon_to"], coords["stop_lat_to"]],
        ]
    )


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "routing.opentripplanner.output_filename",
        "public_transit.chevelus.gtfs_zipfile",
        "public_transit.chevelus.route_id",
        "public_transit.chevelus.output_filename",
    ]
    check_keys(config, mandatory_keys)

    df = read_pt_itineraries(
        config["routing"]["opentripplanner"]["output_filename"],
        config["public_transit"]["chevelus"]["route_id"],
    )
    if not config["public_transit"]["chevelus"].get("all_flows", False):
        assert "run_directory" in config, "Missing key `run_directory` in config"
        df = filter_pt_trips(df, config["run_directory"], 2)
    df = df.collect()
    connections = get_connections(df, config["public_transit"]["chevelus"]["gtfs_zipfile"])
    df = filter_out(df, connections, config["public_transit"]["chevelus"])
    flows = get_flows(df, connections)
    connections = connections.join(flows, on=["route_id", "from", "to"])
    gdf = to_geopandas(connections)
    print("Saving output file...")
    metro_io.save_geodataframe(gdf, config["public_transit"]["chevelus"]["output_filename"])
