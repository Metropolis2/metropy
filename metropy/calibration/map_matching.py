import os
import time

from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig
from fmm import GPSConfig, ResultConfig

import metropy.utils.io as metro_io


def get_graph(filename: str, crs: str, tmp_dir: str):
    """Save the road network as a shapefile representing the graph, readable by Fast Map Matching.
    Returns the filename of the saved file.
    """
    print("Reading edges")
    gdf = metro_io.read_geodataframe(filename)
    gdf.set_index("id", inplace=True, drop=True)
    gdf.sort_index(inplace=True)
    gdf.to_crs(crs, inplace=True)
    gdf = gdf[["source", "target", "geometry"]].copy()
    output_filename = os.path.join(tmp_dir, "fmm_graph.shp")
    print("Saving graph")
    gdf.to_file(output_filename, driver="Shapefile")
    return output_filename


def get_trajectories(filename: str, crs: str, tmp_dir: str):
    """Save the routing results as a shapefile, readable by Fast Map Matching.
    Returns the filename of the saved file.
    """
    print("Reading API results")
    gdf = metro_io.read_geodataframe(filename)
    gdf.sort_values("id", inplace=True)
    gdf.to_crs(crs, inplace=True)
    gdf = gdf[["id", "geometry"]].copy()
    output_filename = os.path.join(tmp_dir, "fmm_gps.shp")
    print("Saving trajectories")
    gdf.to_file(output_filename, driver="Shapefile")
    return output_filename


def run_fmm(graph_filename: str, gps_filename: str, config: dict):
    network = Network(graph_filename)
    graph = NetworkGraph(network)
    model = STMATCH(network, graph)

    input_config = GPSConfig()
    input_config.file = gps_filename
    input_config.id = "id"
    input_config.geom = "geometry"

    result_config = ResultConfig()
    result_config.file = config["output_file"]
    result_config.output_config.write_offset = False
    result_config.output_config.write_error = True
    result_config.output_config.write_opath = False
    result_config.output_config.write_cpath = True
    result_config.output_config.write_ogeom = False
    result_config.output_config.write_mgeom = False

    stmatch_config = STMATCHConfig(
        config["nb_candidates"], config["radius"], config["gps_error"], config.get("factor", 0.05)
    )

    status = model.match_gps_file(input_config, result_config, stmatch_config)
    print(status)


if __name__ == "__main__":
    from metropy.config import read_config

    config = read_config()
    for arg in ("clean_edges_file", "crs", "tmp_directory", "calibration.map_matching"):
        if not arg in config:
            raise Exception(f"Missing key `{arg}` in config")
    fmm_config = config["calibration.map_matching"]
    for arg in ("output_file", "nb_candidates", "gps_error", "radius"):
        if not arg in fmm_config:
            raise Exception(f"Missing key `calibration.map_matching.{arg}` in config")

    t0 = time.time()

    graph_filename = get_graph(config["clean_edges_file"], config["crs"], config["tmp_directory"])
    gps_filename = get_trajectories(
        config["calibration.tomtom"]["output_file"], config["crs"], config["tmp_directory"]
    )
    try:
        run_fmm(graph_filename, gps_filename, config["calibration.map_matching"])
    except:
        pass
    finally:
        # Delete temporary files.
        os.remove(graph_filename)
        os.remove(gps_filename)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
