import os
import time

import numpy as np
import geopandas as gpd
import osmium
from shapely.geometry import LineString
from matplotlib.cm import Set3

import metropy.utils.mpl as mpl
import metropy.utils.io as metro_io


def import_network(config: dict, output_file: str, crs: str) -> gpd.GeoDataFrame:
    """This function takes as input a .osm.pbf file of an area and writes a GeoDataFrame
    representing the edges of the road network.
    The GeoDataFrame has the following columns:
    - `geometry` (EPSG:4326)
    - `osm_id`
    - `source`
    - `target`
    - `length` (m)
    - `name` (can be null)
    - `road_type` (id of the corresponding OSM highway tag)
    """
    t0 = time.time()
    # Check if the `osm` part of the config is valid.
    # It must contain the following keys-value pairs:
    # - `input_file`: path to the `.osm.pbf` file,
    # - `highways`: list of highway tags to import,
    input_file = config.get("input_file")
    if input_file is None:
        raise Exception("Missing key `osm.input_file` in config")
    if not os.path.exists(input_file):
        raise Exception(f"OSM input file not found:\n`{input_file}`")
    highways = config.get("highways")
    if not isinstance(highways, list):
        raise Exception("Invalid or missing array `osm.highways` in config")

    print("Finding nodes...")
    node_reader = NodeReader(highways)
    node_reader.apply_file(input_file, locations=True, idx="flex_mem")

    print("Reading edges...")
    edge_reader = EdgeReader(node_reader.node_ids, node_reader.edge_ids)
    edge_reader.apply_file(input_file, locations=True, idx="flex_mem")

    print("Post-processing...")
    edge_reader.post_process(crs)
    print("Writing edges...")
    edge_reader.write_edges(output_file)
    print("Done!")
    print("Total running time: {:.2f} seconds".format(time.time() - t0))
    return edge_reader.edges_df


def print_stats(gdf: gpd.GeoDataFrame):
    print("Printing stats")
    nb_nodes = len(set(gdf["source"]).union(set(gdf["target"])))
    print(f"Number of nodes: {nb_nodes:,}")
    nb_edges = len(gdf)
    print(f"Number of edges: {nb_edges:,}")
    tot_length = gdf["length"].sum() / 1e3
    print(f"Total edge length (km): {tot_length:,.3f}")


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


def valid_way(way, highways: set[str]) -> bool:
    """Returns True if the way is a valid way to consider."""
    has_access = not "access" in way.tags or way.tags["access"] != "private"
    return (
        has_access
        and len(way.nodes) > 1
        and way.tags.get("highway") in highways
        and not way.tags.get("area") == "yes"
    )


class NodeReader(osmium.SimpleHandler):
    def __init__(self, highways):
        super().__init__()
        # Set of highway tags to be considered.
        self.highways = set(highways)
        # Ids of the nodes explored.
        self.nodes_explored = set()
        # Ids of all the nodes in the final graph.
        self.node_ids = set()
        # Ids of all the edges in the final graph.
        self.edge_ids = set()

    def way(self, way):
        if not valid_way(way, self.highways):
            # Only consider valid highways.
            return
        self.edge_ids.add(way.id)
        # Always add source and target node to the final graph.
        self.node_ids.add(way.nodes[0].ref)
        self.node_ids.add(way.nodes[-1].ref)
        # Add the other nodes if they were already explored, i.e., they
        # intersect with another valid highway.
        for i in range(1, len(way.nodes) - 1):
            node = way.nodes[i]
            if node.ref in self.nodes_explored:
                self.node_ids.add(node.ref)
            else:
                self.nodes_explored.add(node.ref)


class EdgeReader(osmium.SimpleHandler):
    def __init__(self, node_ids, edge_ids):
        super().__init__()
        # Ids of the nodes in the final graph.
        self.node_ids = node_ids
        # Ids of the edges in the final graph.
        self.edge_ids = edge_ids
        # List of edges in the final graph, with their description.
        self.edges: list[dict] = list()

    def way(self, way):
        if not way.id in self.edge_ids:
            return

        road_type = way.tags["highway"]

        name = (
            way.tags.get("name", "") or way.tags.get("addr:street", "") or way.tags.get("ref", "")
        )
        if len(name) > 50:
            name = name[:47] + "..."

        for i, node in enumerate(way.nodes):
            if node.ref in self.node_ids:
                source = i
                break
        else:
            # No node of the way is in the nodes.
            print("Error: Found a valid way ({}) with no valid node".format(way.id))
            return

        j = source + 1
        for i, node in enumerate(list(way.nodes)[j:]):
            if node.ref in self.node_ids:
                target = j + i
                self.add_edge(
                    way,
                    source,
                    target,
                    name,
                    road_type,
                )
                source = target

    def add_edge(self, way, source, target, name, road_type):
        source_id = way.nodes[source].ref
        target_id = way.nodes[target].ref
        if source_id == target_id:
            # Self-loop.
            return
        # Create a geometry of the road.
        coords = list()
        for i in range(source, target + 1):
            node = way.nodes[i]
            if node.location.valid():
                coords.append((node.lon, node.lat))
        geometry = LineString(coords)
        self.edges.append(
            {
                "geometry": geometry,
                "name": name,
                "road_type": road_type,
                "source": source_id,
                "target": target_id,
                "osm_id": way.id,
            }
        )
        # Add backward edge.
        self.edges.append(
            {
                "geometry": geometry.reverse(),
                "name": name,
                "road_type": road_type,
                "source": target_id,
                "target": source_id,
                "osm_id": way.id,
            }
        )

    def post_process(self, metric_crs: str):
        edges = gpd.GeoDataFrame(self.edges, crs="EPSG:4326")
        # Compute length.
        edges["length"] = edges.geometry.to_crs(metric_crs).length
        print("Number of edges: {}".format(len(edges)))
        edges = edges.loc[
            :,
            [
                "geometry",
                "source",
                "target",
                "length",
                "osm_id",
                "name",
                "road_type",
            ],
        ].copy()
        edges["id"] = np.arange(len(edges))
        self.edges_df = edges

    def write_edges(self, output_file):
        metro_io.save_geodataframe(self.edges_df, output_file)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = ["osm_walk", "raw_walk_edges_file", "crs"]
    check_keys(config, mandatory_keys)
    gdf = import_network(config["osm_walk"], config["raw_walk_edges_file"], config["crs"])

    if config["osm_walk"].get("print_stats", False):
        print_stats(gdf)
    if config["osm_walk"].get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "road_network.osm_walk")
        plot_variables(gdf, graph_dir)
