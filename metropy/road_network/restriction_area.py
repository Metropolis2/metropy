import time
import os

import shapely
import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io

# Path to the file with the ZFE geometry.
ZFE_FILE = "./data/ZFE/zfe.geojson"
# Path to the file where the edges are stored.
EDGE_FILE = "./output/osm_network/edges.pickle"
# Path to the directory where edges should be stored.
OUTPUT_DIR = "./output/osm_network/"
# CRS to use for the operations.
CRS = "EPSG:2154"


def read_restriction_area(filename: str, crs: str):
    print("Reading restriction area...")
    gdf = metro_io.read_geodataframe(filename)
    gdf.to_crs(crs, inplace=True)
    geom = gdf.geometry.unary_union
    # TODO: Replace with `gdf.geometry.union_all` when updating to GeoPandas 1.x.y.
    return geom


def read_edges(filename: str, crs: str):
    print("Reading edges...")
    edges = metro_io.read_geodataframe(filename)
    edges.to_crs(crs, inplace=True)
    return edges


def flag_edges(edges: gpd.GeoDataFrame, area: shapely.Geometry):
    print("Finding edges inside the restriction area...")
    inside_area = edges.geometry.within(area)
    print(
        "Number of edges within the area: {:,} ({:.2%} of total)".format(
            inside_area.sum(), inside_area.sum() / len(edges)
        )
    )
    l = edges.loc[inside_area, "length"].sum() / 1000
    print(
        "Length of edges within the area: {:,.0f} km ({:.2%} of total)".format(
            l, l / (edges["length"].sum() / 1000)
        )
    )
    df = pl.DataFrame(
        {
            "edge_id": edges["edge_id"],
            "in_area": pl.Series(inside_area),
        }
    )
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "crs",
        "road_network.restriction_area.area_filename",
        "road_network.restriction_area.output_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    area = read_restriction_area(
        config["road_network"]["restriction_area"]["area_filename"], config["crs"]
    )

    edges = read_edges(config["clean_edges_file"], config["crs"])

    df = flag_edges(edges, area)

    print("Writing output file...")
    metro_io.save_dataframe(df, config["road_network"]["restriction_area"]["output_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
