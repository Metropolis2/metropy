import os
import time

import numpy as np
import geopandas as gpd
import osmium
from osmium.geom import WKBFactory
import pyproj
from shapely.ops import transform
from shapely.geometry import LineString, Point
from shapely.prepared import prep
from matplotlib import colormaps

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
    - `speed` (km/h, can be null)
    - `length` (m)
    - `lanes` (can be null)
    - `urban` (bool, edge is in an urban area)
    - `name` (can be null)
    - `road_type` (id of the corresponding OSM highway tag)
    """
    t0 = time.time()
    # Check if the `osm` part of the config is valid.
    # It must contain the following keys-value pairs:
    # - `input_file`: path to the `.osm.pbf` file,
    # - `highways`: list of highway tags to import,
    # - `urban_landuse`: list of landuse tags to be considered as urban areas (optional).
    input_file = config.get("input_file")
    if input_file is None:
        raise Exception("Missing key `osm.input_file` in config")
    if not os.path.exists(input_file):
        raise Exception(f"OSM input file not found:\n`{input_file}`")
    highways = config.get("highways")
    if not isinstance(highways, list):
        raise Exception("Invalid or missing array `osm.highways` in config")
    urban_landuses = config.get("urban_landuse")
    if not isinstance(urban_landuses, list):
        print("Warning: Invalid or missing array `osm.urban_landuse` in config")
        print("The urban column will not be created")

    print("Finding nodes...")
    node_reader = NodeReader(highways)
    node_reader.apply_file(input_file, locations=True, idx="flex_mem")

    if config.get("print_stats", False):
        print("Number of traffic signals detected: {}".format(len(node_reader.traffic_signal_ids)))
        n = len(node_reader.traffic_signal_ids) - sum(
            1 for v in node_reader.traffic_signal_ids.values() if v is None
        )
        print("Number of traffic signals with direction: {}".format(n))
        print("Number of stop signs detected: {}".format(len(node_reader.stop_signs_ids)))
        n = len(node_reader.stop_signs_ids) - sum(
            1 for v in node_reader.stop_signs_ids.values() if v is None
        )
        print("Number of stop signs with direction: {}".format(n))
        print("Number of give_way signs detected: {}".format(len(node_reader.give_way_signs_ids)))
        n = len(node_reader.give_way_signs_ids) - sum(
            1 for v in node_reader.give_way_signs_ids.values() if v is None
        )
        print("Number of give_way signs with direction: {}".format(n))

    print("Reading edges...")
    edge_reader = EdgeReader(
        node_reader.node_ids,
        node_reader.edge_ids,
        node_reader.traffic_signal_ids,
        node_reader.stop_signs_ids,
        node_reader.give_way_signs_ids,
    )
    edge_reader.apply_file(input_file, locations=True, idx="flex_mem")

    if urban_landuses is None:
        urban_area = None
    else:
        print("Finding urban areas...")
        area_reader = UrbanAreasReader(urban_landuses)
        area_reader.apply_file(input_file, locations=True, idx="flex_mem")
        urban_area = area_reader.get_urban_area()

        # Buffer the urban areas by 50 meters to capture all nearby roads.
        urban_area = buffer(urban_area, 50, crs)
        urban_area = prep(urban_area)

    print("Post-processing...")
    edge_reader.post_process(urban_area, crs)
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
    speed_na_count = gdf["speed"].isna().sum()
    print(f"Number of null values for speed: {speed_na_count:,} ({speed_na_count / nb_edges:.1%})")
    lanes_na_count = gdf["lanes"].isna().sum()
    print(f"Number of null values for lanes: {lanes_na_count:,} ({lanes_na_count / nb_edges:.1%})")


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
        colors=colormaps["Set3"],
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
        colors=colormaps["Set3"],
    )
    fig.savefig(os.path.join(graph_dir, "road_type_pie_length_weights.pdf"))


def valid_way(way, highways: set[str]) -> bool:
    """Returns True if the way is a valid way to consider."""
    has_access = not "access" in way.tags or way.tags["access"] in (
        "yes",
        "permissive",
        "destination",
    )
    return (
        has_access
        and len(way.nodes) > 1
        and way.tags.get("highway") in highways
        and not way.tags.get("area") == "yes"
    )


def is_urban_area(area, urban_landuses: set[str]):
    """Returns True if the area is an urban area."""
    return area.tags.get("landuse") in urban_landuses and (area.num_rings()[0] > 0)


class UrbanAreasReader(osmium.SimpleHandler):
    def __init__(self, urban_landuses):
        super().__init__()
        # Set of landuse tags to be considered as urban areas.
        self.urban_landuses = urban_landuses
        self.wkb_factory = WKBFactory()
        self.areas_wkb = list()

    def area(self, area):
        if not is_urban_area(area, self.urban_landuses):
            return
        self.handle_area(area)

    def handle_area(self, area):
        self.areas_wkb.append(self.wkb_factory.create_multipolygon(area))

    def get_urban_area(self):
        polygons = gpd.GeoSeries.from_wkb(self.areas_wkb)
        #  gdf = gpd.GeoDataFrame(geometry=polygons, crs="epsg:4326")
        #  gdf.to_parquet("output/road_network/urban_areas.parquet")
        return polygons.unary_union


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
        # Ids of the nodes with traffic lights and their direction (if given).
        self.traffic_signal_ids = dict()
        # Ids of the nodes with stop signs and their direction (if given).
        self.stop_signs_ids = dict()
        # Ids of the nodes with give_way signs and their direction (if given).
        self.give_way_signs_ids = dict()

    def node(self, node):
        hw_tag = node.tags.get("highway")
        if hw_tag == "traffic_signals":
            self.traffic_signal_ids[node.id] = node.tags.get(
                "traffic_signals:direction"
            ) or node.tags.get("direction")
        elif hw_tag == "stop":
            self.stop_signs_ids[node.id] = node.tags.get("direction")
        elif hw_tag == "give_way":
            self.give_way_signs_ids[node.id] = node.tags.get("direction")

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
    def __init__(self, node_ids, edge_ids, traffic_signal_ids, stop_signs_ids, give_way_signs_ids):
        super().__init__()
        # Ids of the nodes in the final graph.
        self.node_ids = node_ids
        # Ids of the edges in the final graph.
        self.edge_ids = edge_ids
        # Ids of the nodes with traffic lights.
        self.traffic_signal_ids = traffic_signal_ids
        # Ids of the nodes with stop signs.
        self.stop_signs_ids = stop_signs_ids
        # Ids of the nodes with give_way signs.
        self.give_way_signs_ids = give_way_signs_ids
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

        has_toll = way.tags.get("toll") == "yes"
        is_roundabout = way.tags.get("junction", "") == "roundabout"
        oneway = way.tags.get("oneway", "no") == "yes" or is_roundabout

        # Find maximum speed if available.
        maxspeed = way.tags.get("maxspeed", "")
        speed = None
        back_speed = None
        if maxspeed == "FR:walk":
            speed = 20
        elif maxspeed == "FR:urban":
            speed = 50
        elif maxspeed == "FR:rural":
            speed = 80
        else:
            try:
                speed = float(maxspeed)
            except ValueError:
                pass
        if not oneway:
            try:
                speed = float(way.tags.get("maxspeed:forward", "0")) or speed
            except ValueError:
                pass
            try:
                back_speed = float(way.tags.get("maxspeed:backward", "0")) or speed
            except ValueError:
                pass

        # Find number of lanes if available.
        lanes = None
        back_lanes = None
        if oneway:
            try:
                lanes = int(way.tags.get("lanes", ""))
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
        else:
            try:
                lanes = (
                    int(way.tags.get("lanes:forward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
            try:
                back_lanes = (
                    int(way.tags.get("lanes:backward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                back_lanes = max(back_lanes, 1)

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
                    oneway,
                    is_roundabout,
                    has_toll,
                    name,
                    road_type,
                    lanes,
                    back_lanes,
                    speed,
                    back_speed,
                )
                source = target

    def add_edge(
        self,
        way,
        source,
        target,
        oneway,
        is_roundabout,
        has_toll,
        name,
        road_type,
        lanes,
        back_lanes,
        speed,
        back_speed,
    ):
        source_id = way.nodes[source].ref
        target_id = way.nodes[target].ref
        if source_id == target_id:
            # Self-loop.
            return
        # Create a geometry of the road.
        coords = list()
        traffic_signals = list()
        has_source_traffic_signals = False
        has_target_traffic_signals = False
        stop_signs = list()
        has_source_stop_sign = False
        has_target_stop_sign = False
        give_way_signs = list()
        has_source_give_way_sign = False
        has_target_give_way_sign = False
        for i in range(source, target + 1):
            node = way.nodes[i]
            if node.location.valid():
                coords.append((node.lon, node.lat))
                if node.ref in self.traffic_signal_ids:
                    direction = self.traffic_signal_ids[node.ref]
                    traffic_signals.append((direction, Point(node.lon, node.lat)))
                elif node.ref in self.stop_signs_ids:
                    direction = self.stop_signs_ids[node.ref]
                    stop_signs.append((direction, Point(node.lon, node.lat)))
                elif node.ref in self.give_way_signs_ids:
                    direction = self.give_way_signs_ids[node.ref]
                    give_way_signs.append((direction, Point(node.lon, node.lat)))
        geometry = LineString(coords)
        if traffic_signals:
            source_geom = Point(coords[0])
            target_geom = Point(coords[-1])
            for direction, point in traffic_signals:
                if direction == "forward":
                    has_target_traffic_signals = True
                elif direction == "backward":
                    has_source_traffic_signals = True
                elif direction == "both":
                    has_source_traffic_signals = True
                    has_target_traffic_signals = True
                else:
                    if oneway:
                        # Oneway road: the traffic signal is for the forward direction.
                        has_target_traffic_signals = True
                    else:
                        # Find if the traffic signal is for the forward or backward direction based
                        # on its distance to the source and to the target.
                        dist_to_source = point.distance(source_geom)
                        dist_to_target = point.distance(target_geom)
                        if dist_to_source < dist_to_target:
                            has_source_traffic_signals = True
                        else:
                            has_target_traffic_signals = True
        if stop_signs:
            source_geom = Point(coords[0])
            target_geom = Point(coords[-1])
            for direction, point in stop_signs:
                if direction == "forward":
                    has_target_stop_sign = True
                elif direction == "backward":
                    has_source_stop_sign = True
                elif direction == "both":
                    has_source_stop_sign = True
                    has_target_stop_sign = True
                else:
                    if oneway:
                        # Oneway road: the stop sign is for the forward direction.
                        has_target_stop_sign = True
                    else:
                        # Find if the stop sign is for the forward or backward direction based
                        # on its distance to the source and to the target.
                        dist_to_source = point.distance(source_geom)
                        dist_to_target = point.distance(target_geom)
                        if dist_to_source < dist_to_target:
                            has_source_stop_sign = True
                        else:
                            has_target_stop_sign = True
        if give_way_signs:
            source_geom = Point(coords[0])
            target_geom = Point(coords[-1])
            for direction, point in give_way_signs:
                if direction == "forward":
                    has_target_give_way_sign = True
                elif direction == "backward":
                    has_source_give_way_sign = True
                elif direction == "both":
                    has_source_give_way_sign = True
                    has_target_give_way_sign = True
                else:
                    if oneway:
                        # Oneway road: the give_way sign is for the forward direction.
                        has_target_give_way_sign = True
                    else:
                        # Find if the give_way sign is for the forward or backward direction based
                        # on its distance to the source and to the target.
                        dist_to_source = point.distance(source_geom)
                        dist_to_target = point.distance(target_geom)
                        if dist_to_source < dist_to_target:
                            has_source_give_way_sign = True
                        else:
                            has_target_give_way_sign = True

        self.edges.append(
            {
                "geometry": geometry,
                "name": name,
                "roundabout": is_roundabout,
                "toll": has_toll,
                "traffic_signals": has_target_traffic_signals,
                "stop_sign": has_target_stop_sign,
                "give_way_sign": has_target_give_way_sign,
                "road_type": road_type,
                "lanes": lanes,
                "speed": speed,
                "source": source_id,
                "target": target_id,
                "osm_id": way.id,
            }
        )

        if not oneway:
            self.edges.append(
                {
                    "geometry": geometry.reverse(),
                    "name": name,
                    "roundabout": is_roundabout,
                    "toll": has_toll,
                    "traffic_signals": has_source_traffic_signals,
                    "stop_sign": has_source_stop_sign,
                    "give_way_sign": has_source_give_way_sign,
                    "road_type": road_type,
                    "lanes": back_lanes,
                    "speed": back_speed,
                    "source": target_id,
                    "target": source_id,
                    "osm_id": way.id,
                }
            )

    def post_process(self, urban_area, metric_crs: str):
        edges = gpd.GeoDataFrame(self.edges, crs="EPSG:4326")

        if not urban_area is None:
            edges["urban"] = [urban_area.contains(geom) for geom in edges.geometry]

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
                "speed",
                "lanes",
                "urban",
                "osm_id",
                "name",
                "roundabout",
                "traffic_signals",
                "stop_sign",
                "give_way_sign",
                "toll",
                "road_type",
            ],
        ].copy()

        edges["id"] = np.arange(len(edges))

        self.edges_df = edges

    def write_edges(self, output_file):
        metro_io.save_geodataframe(self.edges_df, output_file)


def buffer(geom, distance, metric_crs):
    wgs84 = pyproj.CRS("EPSG:4326")
    metric_crs = pyproj.CRS(metric_crs)
    project = pyproj.Transformer.from_crs(wgs84, metric_crs, always_xy=True).transform
    inverse_project = pyproj.Transformer.from_crs(metric_crs, wgs84, always_xy=True).transform
    metric_geom = transform(project, geom)
    buffered_geom = metric_geom.buffer(distance).simplify(0, preserve_topology=False)
    geom = transform(inverse_project, buffered_geom)
    return geom


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "osm",
        "raw_edges_file",
        "crs",
    ]
    check_keys(config, mandatory_keys)
    gdf = import_network(config["osm"], config["raw_edges_file"], config["crs"])

    if config["osm"].get("print_stats", False):
        print_stats(gdf)
    if config["osm"].get("output_graphs", False):
        if not "graph_directory" in config:
            raise Exception("Missing key `graph_directory` in config")
        graph_dir = os.path.join(config["graph_directory"], "road_network.osm")
        plot_variables(gdf, graph_dir)
