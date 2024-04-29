import os
import time

import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from matplotlib.cm import Set3

import metropy.utils.mpl as mpl


def postprocess(config: dict, input_file: str, output_file: str):
    """Reads a GeoDataFrame of edges and performs various operations to make the data ready to use
    with METROPOLIS2.
    Saves the results to the given output file.
    """
    t0 = time.time()
    if not os.path.exists(input_file):
        raise Exception(f"Raw edges file not found:\n`{input_file}`")

    if not "default_nb_lanes" in config:
        raise Exception("Missing key `default_nb_lanes` in config")
    if not "default_capacity" in config:
        raise Exception("Missing key `default_capacity` in config")
    if not "default_speed" in config:
        raise Exception("Missing key `default_speed` in config")
    gdf = read_edges(input_file)
    gdf = clean(gdf, config)
    save(gdf, output_file)
    print("Total running time: {:.2f} seconds".format(time.time() - t0))
    return gdf


def read_edges(input_file):
    if input_file.endswith(".parquet"):
        gdf = gpd.read_parquet(input_file)
    else:
        gdf = gpd.read_file(input_file)
    columns = [
        "geometry",
        "source",
        "target",
        "length",
        "speed",
        "lanes",
        "road_type",
    ]
    for col in columns:
        if not col in gdf.columns:
            print("Error: Missing column {}".format(col))
    return gdf


def set_default_values(gdf, config):
    # Set default speeds.
    if "urban" in gdf.columns:
        # Set default speeds based on urban vs rural areas.
        urban_speeds = config["default_speed"].get("urban")
        if not isinstance(urban_speeds, dict):
            raise Exception(
                "Missing or invalid table `postprocess_network.default_speed.urban` in config"
            )
        rural_speeds = config["default_speed"].get("rural")
        if not isinstance(rural_speeds, dict):
            raise Exception(
                "Missing or invalid table `postprocess_network.default_speed.rural` in config"
            )
        urban_speeds = pd.DataFrame(
            list(urban_speeds.values()),
            index=list(urban_speeds.keys()),
            columns=["urban_speed"],
        )
        rural_speeds = pd.DataFrame(
            list(rural_speeds.values()),
            index=list(rural_speeds.keys()),
            columns=["rural_speed"],
        )
        default_speeds = pd.concat((urban_speeds, rural_speeds), axis=1)
        gdf = gdf.merge(default_speeds, left_on="road_type", right_index=True, how="left")
        gdf.loc[gdf["speed"].isna() & gdf["urban"], "speed"] = gdf["urban_speed"]
        gdf.loc[gdf["speed"].isna() & ~gdf["urban"], "speed"] = gdf["rural_speed"]
        gdf = gdf.drop(columns=["urban_speed", "rural_speed"])
    else:
        speeds = config["default_speed"]
        if not isinstance(speeds, dict):
            raise Exception("Invalid table `postprocess_network.default_speed` in config")
        gdf["speed"] = gdf["speed"].fillna(gdf["road_type"].map(speeds))
    # Set default number of lanes.
    nb_lanes = config["default_nb_lanes"]
    if not isinstance(nb_lanes, dict):
        raise Exception("Invalid table `postprocess_network.default_nb_lanes` in config")
    gdf["lanes"] = gdf["lanes"].fillna(gdf["road_type"].map(nb_lanes))
    # Set default bottleneck capacity.
    capacities = config["default_capacity"]
    if not isinstance(capacities, dict):
        raise Exception("Invalid table `postprocess_network.default_capacity` in config")
    if "capacity" in gdf.columns:
        gdf["capacity"] = gdf["capacity"].fillna(gdf["road_type"].map(capacities))
    else:
        gdf["capacity"] = gdf["road_type"].map(capacities)
    return gdf


def remove_duplicates(gdf):
    """Remove the duplicates edges, keeping in order of priority the one in the main graph, with the
    largest capacity and with smallest free-flow travel time."""
    print("Removing duplicate edges")
    n0 = len(gdf)
    l0 = gdf["length"].sum()
    # Sort the dataframe.
    gdf["tt"] = gdf["length"] / (gdf["speed"] / 3.6)
    gdf.sort_values(["capacity", "tt"], ascending=[False, True], inplace=True)
    gdf.drop(columns="tt", inplace=True)
    # Drop duplicates.
    gdf.drop_duplicates(subset=["source", "target"], inplace=True)
    n1 = len(gdf)
    if n0 > n1:
        l1 = gdf["length"].sum()
        print("Warning: discarding {} edges duplicated".format(n0 - n1))
        print("Number of edges removed: {} ({:.2%})".format(n0 - n1, (n0 - n1) / n0))
        print("Edge length removed (m): {:.0f} ({:.2%})".format(l0 - l1, (l0 - l1) / l0))
    return gdf


def select_connected(gdf):
    print("Building graph...")
    G = nx.DiGraph()
    G.add_edges_from(
        map(
            lambda v: (v[0], v[1]),
            gdf[["source", "target"]].values,
        )
    )
    # Keep only the nodes of the largest strongly connected component.
    nodes = max(nx.strongly_connected_components(G), key=len)
    if len(nodes) < G.number_of_nodes():
        print(
            "Warning: discarding {} nodes disconnected from the largest graph component".format(
                G.number_of_nodes() - len(nodes)
            )
        )
        n0 = len(gdf)
        l0 = gdf["length"].sum()
        gdf = gdf.loc[gdf["source"].isin(nodes) & gdf["target"].isin(nodes)].copy()
        n1 = len(gdf)
        l1 = gdf["length"].sum()
        print("Number of edges removed: {} ({:.2%})".format(n0 - n1, (n0 - n1) / n0))
        print("Edge length removed (m): {:.0f} ({:.2%})".format(l0 - l1, (l0 - l1) / l0))
    return gdf


def reindex(gdf):
    gdf["id"] = np.arange(len(gdf))
    return gdf


def check(gdf, config):
    gdf["lanes"] = gdf["lanes"].clip(config.get("min_nb_lanes", 1))
    gdf["length"] = gdf["length"].clip(config.get("min_length", 0.0))
    gdf["speed"] = gdf["speed"].clip(config.get("min_speed", 1e-4))
    # Count number of incoming / outgoing edges for the source / target node.
    target_counts = gdf["target"].value_counts()
    source_counts = gdf["source"].value_counts()
    gdf = gdf.merge(
        target_counts.rename("target_incomings"), left_on="target", right_index=True, how="left"
    )
    gdf = gdf.merge(
        target_counts.rename("source_incomings"), left_on="source", right_index=True, how="left"
    )
    gdf = gdf.merge(
        source_counts.rename("target_outgoings"), left_on="target", right_index=True, how="left"
    )
    gdf = gdf.merge(
        source_counts.rename("source_outgoings"), left_on="source", right_index=True, how="left"
    )
    return gdf


def clean(gdf, config):
    gdf = set_default_values(gdf, config)
    gdf = remove_duplicates(gdf)
    if config.get("ensure_connected", True):
        gdf = select_connected(gdf)
    if config.get("reindex", False):
        gdf = reindex(gdf)
    gdf = check(gdf, config)
    return gdf


def save(gdf, output_file):
    print("Saving post-processed edges")
    directory = os.path.dirname(output_file)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if output_file.endswith("parquet"):
        gdf.to_parquet(output_file)
    elif output_file.endswith("geojson"):
        gdf.to_file(output_file, driver="GeoJSON")
    elif output_file.endswith("fgb"):
        gdf.to_file(output_file, driver="FlatGeobuf")
    elif output_file.endswith("shp"):
        gdf.to_file(output_file, driver="Shapefile")
    else:
        raise Exception(f"Unsupported format for output file: `{output_file}`")


def print_stats(gdf: gpd.GeoDataFrame):
    print("Printing stats")
    nb_nodes = len(set(gdf["source"]).union(set(gdf["target"])))
    print(f"Number of nodes: {nb_nodes:,}")
    nb_edges = len(gdf)
    print(f"Number of edges: {nb_edges:,}")
    nb_urbans = gdf["urban"].sum()
    print(f"Number of urban edges: {nb_urbans:,} ({nb_urbans / nb_edges:.1%})")
    nb_rurals = nb_edges - nb_urbans
    print(f"Number of rural edges: {nb_rurals:,} ({nb_rurals / nb_edges:.1%})")
    nb_roundabouts = gdf["roundabout"].sum()
    print(f"Number of roundabout edges: {nb_roundabouts:,} ({nb_roundabouts / nb_edges:.1%})")
    nb_traffic_signals = gdf["traffic_signals"].sum()
    print(
        f"Number of edges with traffic signals: {nb_traffic_signals:,} ({nb_traffic_signals / nb_edges:.1%})"
    )
    nb_stop_signs = gdf["stop_sign"].sum()
    print(f"Number of edges with stop sign: {nb_stop_signs:,} ({nb_stop_signs / nb_edges:.1%})")
    nb_give_way_signs = gdf["give_way_sign"].sum()
    print(
        f"Number of edges with give_way sign: {nb_give_way_signs:,} ({nb_give_way_signs / nb_edges:.1%})"
    )
    tot_length = gdf["length"].sum() / 1e3
    print(f"Total edge length (km): {tot_length:,.3f}")
    urban_length = gdf.loc[gdf["urban"], "length"].sum() / 1e3
    print(f"Total urban edge length (km): {urban_length:,.3f} ({urban_length / tot_length:.1%})")
    rural_length = tot_length - urban_length
    print(f"Total rural edge length (km): {rural_length:,.3f} ({rural_length / tot_length:.1%})")


def plot_variables(gdf: gpd.GeoDataFrame, graph_dir: str):
    print("Generating graphs of the variables")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    # Length distribution hist.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = np.logspace(np.log(gdf["length"].min()), np.log(gdf["length"].max()), 50, base=np.e)
    ax.hist(gdf["length"], bins=bins, density=True, color=mpl.CMP(0))
    ax.set_xscale("log")
    ax.set_xlabel("Length (meters, log scale)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "length_distribution.pdf"))
    # Speed distribution bar plot.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = np.arange(
        np.floor(gdf["speed"].min() / 5.0) * 5.0 - 2.5,
        np.ceil(gdf["speed"].max() / 5.0) * 5.0 + 2.5 + 1.0,
        5.0,
    )
    ax.hist(gdf["speed"], bins=bins, density=True, color=mpl.CMP(0))
    ax.set_xlabel("Speed limit (km/h)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "speed_distribution.pdf"))
    # Speed distribution bar plot, weighted by length.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = np.arange(
        np.floor(gdf["speed"].min() / 5.0) * 5.0 - 2.5,
        np.ceil(gdf["speed"].max() / 5.0) * 5.0 + 2.5 + 1.0,
        5.0,
    )
    ax.hist(gdf["speed"], bins=bins, density=True, weights=gdf["length"], color=mpl.CMP(0))
    ax.set_xlabel("Speed limit (km/h)")
    ax.set_ylabel("Density (weighted by edge length)")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "speed_distribution_length_weights.pdf"))
    # Lanes distribution bar plot.
    fig, ax = mpl.get_figure(fraction=0.8)
    mask = ~gdf["lanes"].isna()
    bins = [0.5, 1.5, 2.5, 3.5, gdf["lanes"].max()]
    bars, _ = np.histogram(gdf.loc[mask, "lanes"], bins=bins)
    bars = bars / mask.sum()
    xs = np.arange(1, 5)
    ax.bar(xs, bars, width=1.0, color=mpl.CMP(0))
    ax.set_xlabel("Number of lanes")
    ax.set_xticks(xs, ["1", "2", "3", "4+"])
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "lanes_distribution.pdf"))
    # Lanes distribution bar plot, weighted by length.
    fig, ax = mpl.get_figure(fraction=0.8)
    bins = [0.5, 1.5, 2.5, 3.5, gdf["lanes"].max()]
    bars, _ = np.histogram(gdf.loc[mask, "lanes"], bins=bins, weights=gdf.loc[mask, "length"])
    bars = bars / gdf.loc[mask, "length"].sum()
    xs = np.arange(1, 5)
    ax.bar(xs, bars, width=1.0, color=mpl.CMP(0))
    ax.set_xlabel("Number of lanes")
    ax.set_xticks(xs, ["1", "2", "3", "4+"])
    ax.set_ylabel("Density (weighted by edge length)")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "lanes_distribution_length_weights.pdf"))
    # Road type chart.
    fig, ax = mpl.get_figure(fraction=0.8)
    road_type_lengths = gdf["road_type"].value_counts().sort_index()
    # Set the road types with a share <= 2 % in a dedicated category.
    pct_threshold = 2 * len(gdf) / 100
    lengths = dict()
    if (road_type_lengths <= pct_threshold).any():
        lengths["other"] = 0
    for key, value in road_type_lengths.to_dict().items():
        if value <= pct_threshold:
            lengths["other"] += value
        else:
            lengths[key] = value
    ax.pie(
        lengths.values(),
        labels=lengths.keys(),
        autopct=lambda p: f"{p:.1f}\\%",
        pctdistance=0.75,
        labeldistance=1.05,
        colors=Set3.colors,
    )
    fig.savefig(os.path.join(graph_dir, "road_type_pie.pdf"))
    # Road type chart, weighted by length.
    fig, ax = mpl.get_figure(fraction=0.8)
    road_type_lengths = gdf.groupby("road_type")["length"].sum().sort_index()
    # Set the road types with a share <= 2 % in a dedicated category.
    pct_threshold = 2 * gdf["length"].sum() / 100
    lengths = dict()
    if (road_type_lengths <= pct_threshold).any():
        lengths["other"] = 0
    for key, value in road_type_lengths.to_dict().items():
        if value <= pct_threshold:
            lengths["other"] += value
        else:
            lengths[key] = value
    ax.pie(
        lengths.values(),
        labels=lengths.keys(),
        autopct=lambda p: f"{p:.1f}\\%",
        pctdistance=0.75,
        labeldistance=1.05,
        colors=Set3.colors,
    )
    fig.savefig(os.path.join(graph_dir, "road_type_pie_length_weights.pdf"))
    # Capacity distribution bar plot.
    fig, ax = mpl.get_figure(fraction=0.8)
    ax.hist(gdf["capacity"], bins=50, density=True, color=mpl.CMP(0))
    ax.set_xlabel("Capacity (PCE/h)")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "capacity_distribution.pdf"))
    # Capacity distribution bar plot, weighted by length.
    fig, ax = mpl.get_figure(fraction=0.8)
    ax.hist(gdf["capacity"], bins=50, density=True, weights=gdf["capacity"], color=mpl.CMP(0))
    ax.set_xlabel("Capacity (PCE/h)")
    ax.set_ylabel("Density (weighted by edge length)")
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "capacity_distribution_length_weights.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config

    config = read_config()
    if not "postprocess_network" in config:
        raise Exception("Missing key `postprocess_network` in config")
    if not "raw_edges_file" in config:
        raise Exception("Missing key `raw_edges_file` in config")
    if not "clean_edges_file" in config:
        raise Exception("Missing key `clean_edges_file` in config")
    gdf = postprocess(
        config["postprocess_network"], config["raw_edges_file"], config["clean_edges_file"]
    )

    if config["postprocess_network"].get("print_stats", False):
        print_stats(gdf)
    if config["postprocess_network"].get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "road_network.postprocess")
        plot_variables(gdf, graph_dir)
