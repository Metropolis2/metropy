import os
import shutil
import time

import polars as pl

from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig
from fmm import GPSConfig, ResultConfig

import metropy.utils.io as metro_io


def get_graph(filename: str, crs: str, tmp_dir: str):
    """Save the road network as a shapefile representing the graph, readable by Fast Map Matching.
    Returns the filename of the saved file.
    """
    print("Reading edges")
    gdf = metro_io.read_geodataframe(filename, columns=["edge_id", "source", "target", "geometry"])
    gdf.set_index("edge_id", inplace=True, drop=True)
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
    gdf = metro_io.read_geodataframe(filename, columns=["id", "geometry"])
    gdf.sort_values("id", inplace=True)
    gdf.to_crs(crs, inplace=True)
    gdf = gdf[["id", "geometry"]].copy()
    df = pl.DataFrame({"id": gdf["id"], "geometry": gdf.geometry.to_wkt()})
    output_filename = os.path.join(tmp_dir, "fmm_gps.csv")
    print("Saving trajectories")
    df.write_csv(output_filename, separator=";")
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
    result_config.output_config.write_error = False
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
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "crs",
        "tmp_directory",
        "calibration.map_matching.output_file",
        "calibration.map_matching.nb_candidates",
        "calibration.map_matching.gps_error",
        "calibration.map_matching.radius",
        "calibration.tomtom.output_file",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()

    graph_filename = get_graph(config["clean_edges_file"], config["crs"], config["tmp_directory"])
    gps_filename = get_trajectories(
        config["calibration"]["tomtom"]["output_file"], config["crs"], config["tmp_directory"]
    )
    try:
        run_fmm(graph_filename, gps_filename, config["calibration"]["map_matching"])
    except Exception as e:
        print(e)
    finally:
        # Delete temporary directory before returning the error.
        try:
            shutil.rmtree(config["tmp_directory"])
        except OSError as e:
            print(e)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
