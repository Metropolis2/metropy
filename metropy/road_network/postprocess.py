import os
import time

import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd

import metropy.utils.mpl as mpl
import metropy.utils.io as metro_io


def postprocess(config: dict, input_file: str, output_file: str, walk: bool):
    """Reads a GeoDataFrame of edges and performs various operations to make the data ready to use
    with METROPOLIS2.
    Saves the results to the given output file.
    """
    t0 = time.time()
    if not os.path.exists(input_file):
        raise Exception(f"Raw edges file not found:\n`{input_file}`")
    gdf = read_edges(input_file, walk)
    gdf = clean(gdf, config, walk)
    metro_io.save_geodataframe(gdf, output_file)
    print("Total running time: {:.2f} seconds".format(time.time() - t0))
    return gdf


def read_edges(input_file: str, walk: bool):
    gdf = metro_io.read_geodataframe(input_file)
    columns = [
        "geometry",
        "source",
        "target",
        "length",
        "road_type",
    ]
    if not walk:
        columns.extend(["speed", "lanes"])
    for col in columns:
        if not col in gdf.columns:
            print("Error: Missing column {}".format(col))
    return gdf


def set_default_values(gdf, config):
    # Set default speeds.
    gdf["default_speed"] = gdf["speed"].isna()
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
        urban_speeds_df = pd.Series(urban_speeds, name="urban_speed")
        rural_speeds_df = pd.Series(rural_speeds, name="rural_speed")
        default_speeds = pd.concat((urban_speeds_df, rural_speeds_df), axis=1)
        gdf = gdf.merge(default_speeds, left_on="road_type", right_index=True, how="left")
        gdf.loc[gdf["default_speed"] & gdf["urban"], "speed"] = gdf["urban_speed"]
        gdf.loc[gdf["default_speed"] & ~gdf["urban"], "speed"] = gdf["rural_speed"]
        gdf = gdf.drop(columns=["urban_speed", "rural_speed"])
    else:
        speeds = config["default_speed"]
        if not isinstance(speeds, dict):
            raise Exception("Invalid table `postprocess_network.default_speed` in config")
        gdf["speed"] = gdf["speed"].fillna(gdf["road_type"].map(speeds))
    assert not gdf["speed"].isna().any()
    # Set default number of lanes.
    gdf["default_lanes"] = gdf["lanes"].isna()
    nb_lanes = config["default_nb_lanes"]
    if not isinstance(nb_lanes, dict):
        raise Exception("Invalid table `postprocess_network.default_nb_lanes` in config")
    gdf["lanes"] = gdf["lanes"].fillna(gdf["road_type"].map(nb_lanes))
    assert not gdf["lanes"].isna().any()
    # Set default bottleneck capacity.
    #  capacities = config["default_capacity"]
    #  if not isinstance(capacities, dict):
    #  raise Exception("Invalid table `postprocess_network.default_capacity` in config")
    #  if "capacity" in gdf.columns:
    #  gdf["capacity"] = gdf["capacity"].fillna(gdf["road_type"].map(capacities))
    #  else:
    #  gdf["capacity"] = gdf["road_type"].map(capacities)
    return gdf


def remove_duplicates(gdf, walk: bool):
    """Remove the duplicates edges, keeping in order of priority the one in the main graph, with the
    largest capacity and with smallest free-flow travel time."""
    print("Removing duplicate edges")
    n0 = len(gdf)
    l0 = gdf["length"].sum()
    # Sort the dataframe.
    if walk:
        gdf.sort_values("length", ascending=True, inplace=True)
    else:
        gdf["tt"] = gdf["length"] / (gdf["speed"] / 3.6)
        gdf.sort_values(["tt"], ascending=[True], inplace=True)
        gdf.drop(columns="tt", inplace=True)
    # Drop duplicates.
    gdf.drop_duplicates(subset=["source", "target"], inplace=True)
    n1 = len(gdf)
    if n0 > n1:
        l1 = gdf["length"].sum()
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
    gdf["edge_id"] = np.arange(len(gdf), dtype=np.uint64)
    return gdf


def check(gdf, config, walk: bool):
    if not walk:
        gdf["lanes"] = gdf["lanes"].clip(config.get("min_nb_lanes", 1))
        gdf["speed"] = gdf["speed"].clip(config.get("min_speed", 1e-4))
    gdf["length"] = gdf["length"].clip(config.get("min_length", 0.0))
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
    # Add oneway column.
    gdf = gdf.merge(
        gdf[["source", "target"]],
        left_on=["source", "target"],
        right_on=["target", "source"],
        how="left",
        indicator="oneway",
        suffixes=("", "_y"),
    )
    gdf.drop(columns=["source_y", "target_y"], inplace=True)
    gdf.drop_duplicates(subset=["edge_id", "source", "target"], inplace=True)
    gdf["oneway"] = (
        gdf["oneway"]
        .cat.remove_unused_categories()
        .cat.rename_categories({"both": False, "left_only": True})
        .astype(bool)
    )
    return gdf


def clean(gdf, config, walk):
    if not walk:
        gdf = set_default_values(gdf, config)
    if config.get("remove_duplicates", False):
        gdf = remove_duplicates(gdf, walk)
    if config.get("ensure_connected", True):
        gdf = select_connected(gdf)
    if config.get("reindex", False):
        gdf = reindex(gdf)
    gdf = check(gdf, config, walk)
    gdf.sort_values("edge_id", inplace=True)
    return gdf


def print_stats(gdf: gpd.GeoDataFrame, walk: bool):
    print("Printing stats")
    nb_nodes = len(set(gdf["source"]).union(set(gdf["target"])))
    print(f"Number of nodes: {nb_nodes:,}")
    nb_edges = len(gdf)
    print(f"Number of edges: {nb_edges:,}")
    if not walk:
        if "urban" in gdf.columns:
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
        nb_tolls = gdf["toll"].sum()
        print(f"Number of edges with toll: {nb_tolls:,} ({nb_tolls / nb_edges:.1%})")
    tot_length = gdf["length"].sum() / 1e3
    print(f"Total edge length (km): {tot_length:,.3f}")
    if not walk and "urban" in gdf.columns:
        urban_length = gdf.loc[gdf["urban"], "length"].sum() / 1e3
        print(
            f"Total urban edge length (km): {urban_length:,.3f} ({urban_length / tot_length:.1%})"
        )
        rural_length = tot_length - urban_length
        print(
            f"Total rural edge length (km): {rural_length:,.3f} ({rural_length / tot_length:.1%})"
        )


def plot_variables(gdf: gpd.GeoDataFrame, graph_dir: str, walk: bool):
    print("Generating graphs of the variables")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    if not walk:
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
        colors=mpl.COLOR_LIST,
    )
    fig.tight_layout()
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
        colors=mpl.COLOR_LIST,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "road_type_pie_length_weights.pdf"))
    #  # Capacity distribution bar plot.
    #  fig, ax = mpl.get_figure(fraction=0.8)
    #  ax.hist(gdf["capacity"], bins=50, density=True, color=mpl.CMP(0))
    #  ax.set_xlabel("Capacity (PCE/h)")
    #  ax.set_ylabel("Density")
    #  fig.tight_layout()
    #  fig.savefig(os.path.join(graph_dir, "capacity_distribution.pdf"))
    #  # Capacity distribution bar plot, weighted by length.
    #  fig, ax = mpl.get_figure(fraction=0.8)
    #  ax.hist(gdf["capacity"], bins=50, density=True, weights=gdf["capacity"], color=mpl.CMP(0))
    #  ax.set_xlabel("Capacity (PCE/h)")
    #  ax.set_ylabel("Density (weighted by edge length)")
    #  fig.tight_layout()
    #  fig.savefig(os.path.join(graph_dir, "capacity_distribution_length_weights.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "raw_edges_file",
        "clean_edges_file",
        "postprocess_network.default_nb_lanes",
        #  "postprocess_network.default_capacity",
        "postprocess_network.default_speed",
    ]
    check_keys(config, mandatory_keys)

    gdf = postprocess(
        config["postprocess_network"],
        config["raw_edges_file"],
        config["clean_edges_file"],
        walk=False,
    )

    if config["postprocess_network"].get("print_stats", False):
        print_stats(gdf, False)
    if config["postprocess_network"].get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "road_network.postprocess")
        plot_variables(gdf, graph_dir, False)

    # Do the same for the walk road network if needed.
    if "postprocess_network_walk" in config:
        print("\nPostprocessing the walking road network...")
        gdf = postprocess(
            config["postprocess_network_walk"],
            config["raw_walk_edges_file"],
            config["clean_walk_edges_file"],
            walk=True,
        )

        if config["postprocess_network_walk"].get("print_stats", False):
            print_stats(gdf, True)
        if config["postprocess_network_walk"].get("output_graphs", False):
            if not "graph_directory" in config:
                raise Exception("Missing key `graph_directory` in config")
            graph_dir = os.path.join(config["graph_directory"], "road_network.postprocess_walk")
            plot_variables(gdf, graph_dir, True)
