import os
import time
import itertools

import polars as pl
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from tqdm import tqdm

import metropy.utils.io as metro_io


def load_data(edges_filename: str, trajectories_filename: str, crs: str):
    print("Reading edges")
    edges = metro_io.read_geodataframe(
        edges_filename, columns=["edge_id", "source", "target", "length", "geometry"]
    )
    edges.to_crs(crs, inplace=True)
    print("Reading API results")
    trajectories = metro_io.read_geodataframe(
        trajectories_filename, columns=["id", "source", "target", "length", "geometry"]
    )
    trajectories.sort_values("id", inplace=True)
    trajectories.to_crs(crs, inplace=True)
    return edges, trajectories


def map_matching(
    edges: gpd.GeoDataFrame,
    trajectories: gpd.GeoDataFrame,
    radius: float,
    rel_length_threshold: float,
):
    print("Preparing matching")
    # Find the unique nodes in the road network graph, with their Point geometries.
    nodes = (
        edges.drop_duplicates(subset=["source"], ignore_index=True)
        .rename(columns={"source": "node_id"})
        .loc[:, ["node_id", "geometry"]]
    )
    nodes.set_geometry(nodes["geometry"].map(lambda g: Point(g.coords[0])), inplace=True)
    # Create dictionaries to find edge_id from source and target and to find edge's length from
    # edge_id.
    edges_df = pl.from_pandas(edges.loc[:, ["edge_id", "source", "target", "length"]])
    edge_ids = {(r["source"], r["target"]): r["edge_id"] for r in edges_df.iter_rows(named=True)}
    edge_lengths = {r["edge_id"]: r["length"] for r in edges_df.iter_rows(named=True)}
    print("Buffering geometries")
    buffered_trajectories = trajectories.buffer(radius, cap_style="square", resolution=16).simplify(
        tolerance=5, preserve_topology=False
    )
    print("Finding nodes contained within the buffered geometries")
    node_matches = pl.DataFrame(
        nodes.sindex.query(buffered_trajectories, predicate="contains").T,
        schema=["id", "node_id"],
    )
    node_matches = node_matches.with_columns(
        pl.col("node_id").replace_strict(
            nodes.index.values, nodes["node_id"].values, return_dtype=pl.Int64
        )
    )
    node_matches = node_matches.group_by("id").agg("node_id")
    # Add trajectories data (source, target and length).
    node_matches = node_matches.join(
        pl.from_pandas(trajectories.drop(columns=["geometry"])), on="id"
    )
    # Filter out routes for which either the origin or destination node is not in the matched nodes.
    node_matches = node_matches.filter(
        pl.col("node_id").list.contains(pl.col("source")),
        pl.col("node_id").list.contains(pl.col("target")),
    )
    node_matches = node_matches.sort("id")
    nodes = nodes.set_index("node_id")
    results = list()
    for row in tqdm(
        node_matches.iter_rows(named=True), total=len(node_matches), desc="Matching", smoothing=0.05
    ):
        node_ids = set(row["node_id"])
        my_edges = edges_df.filter(
            pl.col("source").is_in(node_ids) & pl.col("target").is_in(node_ids)
        ).select("source", "target", "length")
        if not row["source"] in my_edges["source"] or not row["target"] in my_edges["target"]:
            # Either source or target is not in the graph.
            continue
        G = nx.DiGraph()
        G.add_weighted_edges_from(my_edges.iter_rows())
        tree = nx.bfs_tree(G, row["source"])
        if not tree.has_node(row["target"]):
            # Source and target are not connected.
            #  assert not nx.has_path(G, row["source"], row["target"])
            continue
        dists = nodes.loc[list(tree.nodes)].distance(trajectories.loc[row["id"], "geometry"])
        func = lambda s, _t, _w: dists[s]
        _, path_nodes = nx.bidirectional_dijkstra(tree, row["source"], row["target"], func)
        path_edges = [edge_ids[(s, t)] for s, t in itertools.pairwise(path_nodes)]
        tot_length = sum(edge_lengths[e] for e in path_edges)
        results.append(
            {
                "id": row["id"],
                "path": path_edges,
                "length": tot_length,
                "length_tomtom": row["length"],
            }
        )
    df = (
        pl.DataFrame(results)
        .with_columns(pl.col("length_tomtom").cast(pl.Float64))
        .with_columns(
            rel_length_diff=(pl.col("length") - pl.col("length_tomtom")) / pl.col("length_tomtom")
        )
        .filter(pl.col("rel_length_diff").abs() <= rel_length_threshold)
    )
    n = len(df)
    s = n / len(trajectories)
    print(f"{n:,} routes were matched (representing {s:.1%} of routes)")
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "crs",
        "calibration.tomtom.output_filename",
        "calibration.map_matching.output_filename",
        "calibration.map_matching.radius",
        "calibration.map_matching.rel_length_threshold",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()

    edges, trajectories = load_data(
        config["clean_edges_file"],
        config["calibration"]["tomtom"]["output_filename"],
        config["crs"],
    )

    df = map_matching(
        edges,
        trajectories,
        config["calibration"]["map_matching"]["radius"],
        config["calibration"]["map_matching"]["rel_length_threshold"],
    )

    print("Saving matches")
    df.write_parquet(config["calibration"]["map_matching"]["output_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
