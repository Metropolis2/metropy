from collections import defaultdict
from zipfile import ZipFile
import os
from itertools import pairwise

import polars as pl
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import metropy.utils.mpl as mpl
import metropy.utils.io as metro_io

# Path to the output file of the PT itineraries with flows.
OUTPUT_FILE = "./output/public_transit/global_flows.parquet"


def read_pt_itineraries(filename: str):
    print("Reading public-transit itineraries...")
    df = metro_io.read_dataframe(filename)
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
    df = df.with_columns(
        pl.col("legs").list.len().alias("nb_legs"),
        (
            pl.col("travel_time")
            - pl.col("legs").list.eval(pl.element().struct.field("travel_time")).list.sum()
        ).alias("waiting_time"),
    )
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


def print_stats_and_plot_graphs(df: pl.DataFrame, gtfs_zipfile: str, graph_dir: str):
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    print(f"Number of trips with itinerary found: {len(df):,}")
    print(f"Number of walk-only trips: {df['walk_only'].sum():,} ({df['walk_only'].mean():.2%})")
    print("Travel time (all trips, min.):\n{}".format((df["travel_time"] / 60).describe()))
    print(
        "Travel time (walk-only trips, min.):\n{}".format(
            df.lazy().filter(pl.col("walk_only")).select(pl.col("travel_time") / 60).describe()
        )
    )
    print(
        "Travel time (PT trips, min.):\n{}".format(
            df.lazy().filter(~pl.col("walk_only")).select(pl.col("travel_time") / 60).describe()
        )
    )
    nb_routes = (
        df.lazy()
        .select(
            pl.col("legs").list.eval(pl.element().struct.field("route_id")).explode().n_unique()
        )
        .collect()
        .item()
    )
    print(f"Number of unique public-transit lines: {nb_routes:,}")
    nb_routes_by_mode = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("route_id"))
            .explode()
            .alias("route_id"),
            pl.col("legs").list.eval(pl.element().struct.field("mode")).explode().alias("mode"),
        )
        .filter(pl.col("route_id").is_not_null())
        .group_by("mode")
        .agg(pl.col("route_id").n_unique())
        .collect()
    )
    print(f"Number of unique public-transit lines by mode:\n{nb_routes_by_mode}")

    modes = (
        df.lazy()
        .filter(~pl.col("walk_only"))
        .select(
            pl.col("legs").list.eval(pl.element().struct.field("mode")).explode().alias("mode"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("travel_time"))
            .explode()
            .alias("travel_time"),
        )
        .group_by("mode")
        .agg(
            pl.len().alias("count"),
            pl.col("travel_time").sum() / 3600,
            (pl.col("travel_time").mean() / 60).alias("mean_tt"),
        )
        .sort("count")
        .collect()
    )
    for row in modes.iter_rows(named=True):
        print(f"Number of {row['mode']} legs: {row['count']:,}")
        print(f"Travel time on {row['mode']} legs (hours): {row['travel_time']:,.0f}")
        print(f"Average travel time on {row['mode']} legs (minutes): {row['mean_tt']:.2f}")
    tot_waiting_time = df.select(pl.col("waiting_time").sum() / 60).item()
    print(
        "Average waiting time per trip with at least 2 legs (minutes): {:.2f}".format(
            tot_waiting_time / len(df.filter(pl.col("nb_legs") > 3))
        )
    )

    # Modes pie chart.
    fig, ax = mpl.get_figure(fraction=0.8)
    ax.pie(
        modes["count"],
        labels=modes["mode"],
        autopct=lambda p: f"{p:.1f}\\%",
        pctdistance=0.75,
        labeldistance=1.05,
        colors=mpl.COLOR_LIST,
    )
    fig.savefig(os.path.join(graph_dir, "leg_modes_pie_chart.pdf"))

    # Modes pie chart, weighted by travel time.
    fig, ax = mpl.get_figure(fraction=0.8)
    ax.pie(
        list(modes["travel_time"]) + [tot_waiting_time / 60],
        labels=list(modes["mode"]) + ["WAIT"],
        autopct=lambda p: f"{p:.1f}\\%",
        pctdistance=0.75,
        labeldistance=1.05,
        colors=mpl.COLOR_LIST,
    )
    fig.savefig(os.path.join(graph_dir, "leg_modes_pie_chart_by_travel_time.pdf"))

    routes = (
        df.lazy()
        .select(
            pl.col("legs").list.eval(pl.element().struct.field("mode")).explode().alias("mode"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("route_id"))
            .explode()
            .alias("route_id"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("travel_time"))
            .explode()
            .alias("travel_time"),
        )
        .filter(pl.col("route_id").is_not_null())
        .group_by("route_id")
        .agg(pl.col("mode").first(), pl.len().alias("count"), pl.col("travel_time").sum() / 3600)
        .sort("count", descending=True)
        .collect()
    )
    gtfs_routes = read_as_dataframe(gtfs_zipfile, "routes.txt")
    routes = routes.join(
        gtfs_routes.select("route_id", pl.col("route_long_name").alias("route_name")),
        on="route_id",
        how="left",
        coalesce=False,
    )
    print("Most used PT lines:")
    for mode_df in routes.partition_by(["mode"]):
        for row in mode_df[:10].iter_rows(named=True):
            print(
                "{} - {}: {:,} ({:,.0f} hours)".format(
                    row["mode"], row["route_name"], row["count"], row["travel_time"]
                )
            )
        mode = mode_df["mode"][0]
        # Line pie chart by mode, weighted by travel time.
        fig, ax = mpl.get_figure(fraction=0.8)
        # Set the lines not in the top 10 or with less than 2% in a dedicated category.
        tot_tt = mode_df.select(pl.col("travel_time").sum()).item()
        tt_threshold = 0.02 * tot_tt
        n = len(mode_df[:10].filter(pl.col("travel_time") >= tt_threshold))
        other_tt = tot_tt - mode_df[:n].select(pl.col("travel_time").sum()).item()
        ax.pie(
            list(mode_df[:n]["travel_time"]) + [other_tt],
            labels=list(mode_df[:n]["route_name"]) + ["Other"],
            autopct=lambda p: f"{p:.0f}\\%",
            pctdistance=0.75,
            labeldistance=1.05,
            colors=mpl.COLOR_LIST,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(graph_dir, f"line_pie_chart_by_travel_time_{mode}.pdf"))

    from_stops = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("from_stop_id"))
            .explode()
            .alias("from_stop_id"),
        )
        .filter(pl.col("from_stop_id").is_not_null())
        .group_by("from_stop_id")
        .agg(pl.len().alias("from_count"))
        .sort("from_count", descending=True)
        .rename({"from_stop_id": "stop_id"})
    )
    to_stops = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("to_stop_id"))
            .explode()
            .alias("to_stop_id"),
        )
        .filter(pl.col("to_stop_id").is_not_null())
        .group_by("to_stop_id")
        .agg(pl.len().alias("to_count"))
        .sort("to_count", descending=True)
        .rename({"to_stop_id": "stop_id"})
    )
    stops = (
        from_stops.join(to_stops, on=pl.col("stop_id"), how="full", coalesce=True)
        .select(
            "stop_id",
            (pl.col("from_count").fill_null(0) + pl.col("to_count").fill_null(0)).alias("count"),
        )
        .sort("count", descending=True)
        .collect()
    )
    gtfs_stops = read_as_dataframe(gtfs_zipfile, "stops.txt").select("stop_id", "stop_name")
    stops = stops.join(
        gtfs_stops,
        on="stop_id",
        how="left",
        coalesce=False,
    )
    print("Most used stops:")
    for row in stops[:10].iter_rows(named=True):
        print("{}: {:,}".format(row["stop_name"], row["count"]))

    # Most popular stop pairs.
    stop_pairs = (
        df.lazy()
        .select(
            pl.col("legs")
            .list.eval(pl.element().struct.field("from_stop_id"))
            .explode()
            .alias("from_stop_id"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("to_stop_id"))
            .explode()
            .alias("to_stop_id"),
            pl.col("legs")
            .list.eval(pl.element().struct.field("travel_time"))
            .explode()
            .alias("travel_time"),
        )
        .filter(pl.col("from_stop_id").is_not_null())
        .group_by(["from_stop_id", "to_stop_id"])
        .agg(pl.len().alias("count"), pl.col("travel_time").sum() / 3600)
        .sort("count", descending=True)
        .collect()
    )
    stop_pairs = (
        stop_pairs.join(
            gtfs_stops, left_on="from_stop_id", right_on="stop_id", how="left", coalesce=False
        )
        .rename({"stop_name": "from_stop_name"})
        .join(gtfs_stops, left_on="to_stop_id", right_on="stop_id", how="left", coalesce=False)
        .rename({"stop_name": "to_stop_name"})
    )
    print("Most used stop-pairs:")
    for row in stop_pairs[:10].iter_rows(named=True):
        print(
            "From {} to {}: {:,}".format(row["from_stop_name"], row["to_stop_name"], row["count"])
        )

    entry_stops = (
        df.lazy()
        .filter(~pl.col("walk_only"))
        .select(pl.col("legs").list.get(1).struct.field("from_stop_id").value_counts())
        .unnest("from_stop_id")
        .rename({"from_stop_id": "stop_id"})
        .sort("count", descending=True)
        .collect()
    )
    entry_stops = entry_stops.join(
        gtfs_stops,
        on="stop_id",
        how="left",
        coalesce=False,
    )
    print("Most used stops at origin:")
    for row in entry_stops[:10].iter_rows(named=True):
        print("{}: {:,}".format(row["stop_name"], row["count"]))

    exit_stops = (
        df.lazy()
        .filter(~pl.col("walk_only"))
        .select(pl.col("legs").list.get(-2).struct.field("to_stop_id").value_counts())
        .unnest("to_stop_id")
        .rename({"to_stop_id": "stop_id"})
        .sort("count", descending=True)
        .collect()
    )
    exit_stops = exit_stops.join(
        gtfs_stops,
        on="stop_id",
        how="left",
        coalesce=False,
    )
    print("Most used stops at destination:")
    for row in exit_stops[:10].iter_rows(named=True):
        print("{}: {:,}".format(row["stop_name"], row["count"]))


def save_connections(df: pl.DataFrame, gtfs_zipfile: str, output_filename: str):
    connections = get_connections(df, gtfs_zipfile)
    flows = get_flows(df, connections)
    connections = connections.join(flows, on=["route_id", "from", "to"])
    gdf = to_geopandas(connections)
    print("Saving output file...")
    metro_io.save_geodataframe(gdf, output_filename)


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
        coalesce=False,
    ).join(
        stops.lazy().rename({col: f"{col}_to" for col in columns}),
        left_on="to",
        right_on="stop_id_to",
        how="left",
        coalesce=False,
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
        routes.lazy(), on="route_id", how="left", coalesce=False
    ).with_columns(pl.col("route_color").str.replace("^", "#"))
    return connections.collect()


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
    pairs = connections.select(pl.col("route_id"), pl.struct("from", "to").alias("pair")).filter(
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
        print(
            "Warning. Cannot find path in route for {} segments (representing {:.2%} of segments)".format(
                i, i / raw_pairs["count"].sum()
            )
        )
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
    gdf = gpd.GeoDataFrame(
        connections.to_pandas(), geometry=linestrings.to_numpy(), crs="EPSG:4326"
    )
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
    return [
        f"{from_stop}->{to_stop}"
        for i, from_stop in enumerate(stop_list[:-2])
        for to_stop in stop_list[i + 2 :]
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
        "public_transit.analyze_flows.gtfs_zipfile",
        "public_transit.analyze_flows.output_filename",
        "graph_directory",
    ]
    check_keys(config, mandatory_keys)
    graph_dir = os.path.join(config["graph_directory"], "public_transit")

    df = read_pt_itineraries(config["routing"]["opentripplanner"]["output_filename"])
    print_stats_and_plot_graphs(
        df, config["public_transit"]["analyze_flows"]["gtfs_zipfile"], graph_dir
    )

    save_connections(
        df,
        config["public_transit"]["analyze_flows"]["gtfs_zipfile"],
        config["public_transit"]["analyze_flows"]["output_filename"],
    )
