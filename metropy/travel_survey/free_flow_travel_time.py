import os
import time
import json
import subprocess

import polars as pl
import geopandas as gpd
from shapely.geometry import Point

import metropy.utils.io as metro_io
import metropy.utils.mpl as mpl


def read_egt(directory: str, crs: str):
    print("Reading EGT...")
    lf = pl.scan_csv(os.path.join(directory, "Format_csv", "Deplacements_semaine.csv"))
    # Select trips between 9 p.m. and 6 a.m.
    lf = lf.filter((pl.col("ORH") >= 21) | (pl.col("DESTH") < 6))
    # Select trips by car.
    lf = lf.filter(pl.col("MODP_H6") == 2)
    # Remove trips with origin = destination.
    lf = lf.filter(pl.col("ORC") != pl.col("DESTC"))
    # Remove trips with travel time larger than 2 hours.
    lf = lf.filter(pl.col("DUREE") <= 120)
    lf = lf.select("DUREE", "ORC", "DESTC")
    df = lf.collect()
    carrs = set(df["ORC"]).union(set(df["DESTC"]))
    gdf = gpd.read_file(
        os.path.join(directory, "carreaux", "carr100m.shp"),
        columns=["IDENT", "X100", "Y100"],
        read_geometry=False,
    )
    gdf = gdf.loc[gdf["IDENT"].isin(carrs)].copy()
    gdf = gdf.set_geometry(gpd.points_from_xy(gdf["X100"], gdf["Y100"], crs="epsg:27561"))
    gdf.to_crs(crs, inplace=True)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    carr_df = pl.from_pandas(gdf[["IDENT", "x", "y"]])
    df = df.join(carr_df, left_on="ORC", right_on="IDENT", how="inner").rename(
        {"x": "x_origin", "y": "y_origin"}
    )
    df = df.join(carr_df, left_on="DESTC", right_on="IDENT", how="inner").rename(
        {"x": "x_destination", "y": "y_destination"}
    )
    df = df.select(
        pl.int_range(pl.len(), dtype=pl.UInt64).alias("id"),
        pl.col("DUREE").alias("travel_time"),
        "x_origin",
        "y_origin",
        "x_destination",
        "y_destination",
    )
    print("Number of trips: {:,}".format(len(df)))
    return df


def read_edges(
    filename: str, crs: str, forbidden_road_types: list | None, edge_penalties_filename: str | None
):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(
        filename,
        columns=["edge_id", "source", "target", "length", "speed", "road_type", "geometry"],
    )
    gdf.to_crs(crs, inplace=True)
    if forbidden_road_types is not None:
        gdf["allow_od"] = ~gdf["road_type"].isin(forbidden_road_types)
    else:
        gdf["allow_od"] = True
    if edge_penalties_filename is not None:
        penalties = metro_io.read_dataframe(edge_penalties_filename)
        if "speed" in penalties.columns:
            print("Using multiplicative penalties")
            gdf = gpd.GeoDataFrame(
                gdf.merge(
                    penalties.select("edge_id", fixed_speed="speed").to_pandas(),
                    on="edge_id",
                    how="left",
                )
            )
            gdf["travel_time"] = gdf["length"] / (gdf["fixed_speed"] / 3.6)
        else:
            gdf["travel_time"] = gdf["length"] / (gdf["speed"] / 3.6)
        if "additive_penalty" in penalties.columns:
            print("Using additive penalties")
            gdf = gpd.GeoDataFrame(
                gdf.merge(
                    penalties.select("edge_id", "additive_penalty").to_pandas(),
                    on="edge_id",
                    how="left",
                )
            )
            gdf["travel_time"] += gdf["additive_penalty"].fillna(0.0)
    return gdf


def find_origin_destination_node(df: pl.DataFrame, edges: gpd.GeoDataFrame):
    print("Creating origin / destination points...")
    # Create a GeoDataFrame of all the road-network nodes.
    nodes = gpd.GeoDataFrame(
        edges.loc[edges.loc[:, "allow_od"]]
        .drop_duplicates(subset=["source"])
        .rename(columns={"source": "node_id"})[["node_id", "geometry"]]
    )
    nodes.set_geometry(nodes.geometry.apply(lambda g: Point(g.coords[0])), inplace=True)
    # Create a GeoDataFrame of all the origin centroids.
    origins = gpd.GeoDataFrame(
        data={"id": df["id"]},
        geometry=gpd.points_from_xy(df["x_origin"], df["y_origin"], crs=edges.crs),
    )
    origins = origins.sjoin_nearest(
        nodes,
        distance_col="dist",
        how="left",
    )
    # Duplicate indices occur when there are two nodes at the same distance.
    origins.drop_duplicates(subset=["id"], inplace=True)
    origins = origins.loc[origins.loc[:, "dist"] < 150.0].copy()
    # Create a GeoDataFrame of all the destination centroids.
    destinations = gpd.GeoDataFrame(
        data={"id": df["id"]},
        geometry=gpd.points_from_xy(df["x_destination"], df["y_destination"], crs=edges.crs),
    )
    destinations = destinations.sjoin_nearest(
        nodes,
        distance_col="dist",
        how="left",
    )
    # Duplicate indices occur when there are two nodes at the same distance.
    destinations.drop_duplicates(subset=["id"], inplace=True)
    destinations = destinations.loc[destinations.loc[:, "dist"] < 150.0].copy()
    df = df.join(pl.from_pandas(origins[["id", "node_id"]]), on="id", how="inner").rename(
        {"node_id": "origin"}
    )
    df = df.join(pl.from_pandas(destinations[["id", "node_id"]]), on="id", how="inner").rename(
        {"node_id": "destination"}
    )
    df = df.select("id", "travel_time", "origin", "destination")
    print("Number of valid trips: {:,}".format(len(df)))
    return df


def prepare_routing(df: pl.DataFrame, edges: gpd.GeoDataFrame, tmp_directory: str):
    print("Saving queries...")
    queries = df.select(
        query_id="id",
        origin="origin",
        destination="destination",
        departure_time=pl.lit(0.0),
    )
    queries.write_parquet(os.path.join(tmp_directory, "queries.parquet"))
    print("Saving graph...")
    edges_df = pl.from_pandas(edges.loc[:, ["edge_id", "source", "target", "travel_time"]])
    # Parallel edges are removed, keeping in priority edges with smallest travel time.
    edges_df.sort("travel_time", descending=False).unique(
        subset=["source", "target"], keep="first"
    ).sort("edge_id").select("edge_id", "source", "target", "travel_time").write_parquet(
        os.path.join(tmp_directory, "edges.parquet")
    )
    print("Saving parameters...")
    parameters = {
        "algorithm": "TCH",
        "output_route": False,
        "input_files": {
            "queries": "queries.parquet",
            "edges": "edges.parquet",
        },
        "output_directory": "output",
        "saving_format": "Parquet",
    }
    with open(os.path.join(tmp_directory, "parameters.json"), "w") as f:
        json.dump(parameters, f)


def run_routing(routing_exec: str, tmp_directory: str):
    parameters_filename = os.path.join(tmp_directory, "parameters.json")
    print("Running routing...")
    subprocess.run(" ".join([routing_exec, parameters_filename]), shell=True)


def load_results(tmp_directory: str):
    print("Reading routing results...")
    return pl.read_parquet(
        os.path.join(tmp_directory, "output", "ea_results.parquet"),
    ).select(
        pl.col("query_id").alias("id"),
        (pl.col("arrival_time") / 60).alias("metropolis_travel_time"),
    )


def analyze_results(df: pl.DataFrame, results: pl.DataFrame, graph_dir: str):
    df = df.rename({"travel_time": "survey_travel_time"}).join(results, on="id", how="inner")
    print("Free-flow travel time (survey):\n{}".format(df["survey_travel_time"].describe()))
    print("Free-flow travel time (METROPOLIS):\n{}".format(df["metropolis_travel_time"].describe()))
    fig, ax = mpl.get_figure(fraction=0.8)
    ax.scatter(
        df["survey_travel_time"],
        df["metropolis_travel_time"],
        color=mpl.CMP(0),
        marker=".",
        alpha=0.5,
    )
    ax.plot([0, 120], [0, 120], color="black")
    ax.set_xlabel("Survey travel time (min.)")
    ax.set_ylabel("METROPOLIS2 travel time (min.)")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "free_flow_travel_time_survey_comparison.pdf"))


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "crs",
        "routing_exec",
        "tmp_directory",
        "travel_survey.directory",
        "travel_survey.survey_type",
        "graph_directory",
    ]
    check_keys(config, mandatory_keys)

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()
    directory = config["travel_survey"]["directory"]
    survey_type = config["travel_survey"]["survey_type"]
    if survey_type == "EGT":
        df = read_egt(directory, config["crs"])
    else:
        raise Exception(f"Error. Unsupported survey type: {survey_type}")

    edges = read_edges(
        config["clean_edges_file"],
        config["crs"],
        config.get("forbidden_road_types"),
        config.get("edge_penalties_file"),
    )

    df = find_origin_destination_node(df, edges)

    prepare_routing(df, edges, config["tmp_directory"])

    run_routing(config["routing_exec"], config["tmp_directory"])

    results = load_results(config["tmp_directory"])

    graph_dir = os.path.join(config["graph_directory"], "travel_survey.free_flow_travel_time")
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    analyze_results(df, results, graph_dir)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
