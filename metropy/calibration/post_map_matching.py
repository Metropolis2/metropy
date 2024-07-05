import time

import polars as pl
import geopandas as gpd

import metropy.utils.io as metro_io


def read_edges(input_file: str, crs: str):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(input_file)
    gdf.to_crs(crs, inplace=True)
    return gdf


def read_tomtom(input_file: str, crs: str):
    print("Reading TomTom requests...")
    gdf = metro_io.read_geodataframe(input_file)
    gdf.to_crs(crs, inplace=True)
    print(f"Number of routes from the requests: {len(gdf)}")
    return gdf


def read_matches(input_file: str):
    print("Reading map matching results...")
    df = pl.read_csv(input_file, separator=";")
    df = df.drop_nulls()
    df = df.with_columns(pl.col("cpath").str.split(",").list.eval(pl.element().cast(pl.UInt64)))
    print(f"Number of routes matched: {len(df)}")
    return df


def clean_od(edges: gpd.GeoDataFrame, tomtom: gpd.GeoDataFrame, matches: pl.DataFrame):
    edge_ids = set(edges["edge_id"])
    assert matches["cpath"].list.eval(pl.element().is_in(edge_ids).all()).list.all().all()
    # Check if the origin and destination from the requests match either the source or target node
    # from the first and last edge of the match.
    print("Cleaning origin / destination...")
    source_and_targets = pl.from_pandas(
        edges.loc[:, ["edge_id", "source", "target"]],
        schema_overrides={"edge_id": pl.UInt64, "source": pl.UInt64, "target": pl.UInt64},
    )
    matches = matches.with_columns(
        pl.col("cpath").list.first().alias("first_edge"),
        pl.col("cpath").list.last().alias("last_edge"),
    )
    matches = matches.join(
        source_and_targets, left_on=pl.col("first_edge"), right_on="edge_id", how="inner"
    ).drop("first_edge")
    matches = matches.rename({"source": "source_first", "target": "target_first"}).drop("edge_id")
    matches = matches.join(
        source_and_targets, left_on=pl.col("last_edge"), right_on="edge_id", how="inner"
    ).drop("last_edge")
    matches = matches.rename({"source": "source_last", "target": "target_last"}).drop("edge_id")
    orig_dest = pl.DataFrame(tomtom.loc[:, ["id", "source", "target"]])
    orig_dest = orig_dest.rename({"source": "origin", "target": "destination"})
    matches = matches.join(orig_dest, on="id", how="inner")
    matches = matches.with_columns(
        (pl.col("source_first") == pl.col("origin")).alias("match_o_s"),
        (pl.col("target_first") == pl.col("origin")).alias("match_o_t"),
        (pl.col("source_last") == pl.col("destination")).alias("match_d_s"),
        (pl.col("target_last") == pl.col("destination")).alias("match_d_t"),
    ).drop("source_first", "target_first", "source_last", "target_last", "origin", "destination")
    #  print("Nb match origin / source: {}".format(matches["match_o_s"].sum()))
    #  print("Nb match origin / target: {}".format(matches["match_o_t"].sum()))
    #  print("Nb match destination / source: {}".format(matches["match_d_s"].sum()))
    #  print("Nb match destination / target: {}".format(matches["match_d_t"].sum()))
    matches = matches.with_columns(
        (
            (pl.col("match_o_s") | pl.col("match_o_t"))
            & (pl.col("match_d_s") | pl.col("match_d_t"))
        ).alias("valid_od")
    )
    n = matches["valid_od"].sum()
    print(f"Matches with valid OD: {n}")
    # Correct the cpath to match TomTom origin / destination if needed.
    matches = matches.with_columns(
        pl.when(pl.col("match_o_t")).then(pl.col("cpath").list.slice(1)).otherwise(pl.col("cpath"))
    ).drop("match_o_s", "match_o_t")
    matches = matches.with_columns(
        pl.when(pl.col("match_d_s"))
        .then(pl.col("cpath").list.slice(0, pl.col("cpath").list.len() - 1))
        .otherwise(pl.col("cpath"))
    ).drop("match_d_s", "match_d_t")
    return matches.filter(pl.col("valid_od")).drop("valid_od")


def clean_paths(edges: gpd.GeoDataFrame, matches: pl.DataFrame):
    print("Cleaning paths...")
    sources = {edge_id: source for edge_id, source in zip(edges["edge_id"], edges["source"])}
    targets = {edge_id: target for edge_id, target in zip(edges["edge_id"], edges["target"])}
    new_paths = list()
    valid_paths = list()
    for path in matches["cpath"]:
        node_pos = dict()
        new_path = list()
        for e in path:
            src = sources[e]
            prev_pos = node_pos.get(src)
            if prev_pos is not None:
                new_path = new_path[:prev_pos]
            node_pos[src] = len(new_path)
            new_path.append(e)
        assert len(new_path) == len(set(new_path))
        for e1, e2 in zip(new_path[:-1], new_path[1:]):
            if targets[e1] != sources[e2]:
                valid_paths.append(False)
                break
        else:
            valid_paths.append(True)
        new_paths.append(new_path)
    matches = matches.with_columns(
        pl.Series(new_paths, dtype=pl.List(pl.UInt64)).alias("cpath"),
        pl.Series(valid_paths).alias("valid_path"),
    )
    n = matches["valid_path"].sum()
    print(f"Matches with valid path: {n}")
    return matches.filter(pl.col("valid_path")).drop("valid_path")


def clean_length(edges: gpd.GeoDataFrame, tomtom: gpd.GeoDataFrame, matches: pl.DataFrame):
    print("Cleaning path length difference...")
    lengths = {edge_id: length for edge_id, length in zip(edges["edge_id"], edges["length"])}
    matches = matches.with_columns(
        pl.col("cpath").list.eval(pl.element().replace(lengths).sum()).list.first().alias("length")
    )
    matches = matches.join(
        pl.from_pandas(tomtom.loc[:, ["id", "length"]]), on="id", how="inner", suffix="_tomtom"
    )
    matches = matches.with_columns(pl.col("length_tomtom").cast(pl.Float64))
    matches = matches.with_columns(
        ((pl.col("length") - pl.col("length_tomtom")) / pl.col("length_tomtom")).alias(
            "rel_length_diff"
        )
    )
    matches = matches.with_columns(
        pl.col("rel_length_diff").is_between(-0.002, 0.002).alias("valid_length_diff")
    )
    n = matches["valid_length_diff"].sum()
    print(f"Matches with correct length difference: {n}")
    return matches.filter(pl.col("valid_length_diff")).drop("rel_length_diff", "valid_length_diff")


def clean_covered(edges: gpd.GeoDataFrame, tomtom: gpd.GeoDataFrame, matches: pl.DataFrame):
    print("Check if matched path is within 100m of TomTom path...")
    print("Building points...")
    edges = edges.loc[edges["edge_id"].isin(matches["cpath"].explode().to_pandas())].copy()
    edge_geoms = edges.set_index("edge_id").geometry
    xs = edge_geoms.apply(lambda geom: geom.coords[0][0])
    ys = edge_geoms.apply(lambda geom: geom.coords[0][1])
    source_points = gpd.GeoSeries(
        gpd.points_from_xy(xs, ys), index=edge_geoms.index, crs=edge_geoms.crs
    )
    source_points_dict = source_points.to_dict()
    tomtom.set_index("id", inplace=True)
    tomtom = tomtom.loc[tomtom.index.isin(matches["id"])]
    print("Computing distances...")
    covered = list()
    for i, (query_id, cpath) in enumerate(zip(matches["id"], matches["cpath"])):
        if (i & (i - 1) == 0) and i > 8:
            print(f"{i} / {len(matches)}")
        points = [source_points_dict[e] for e in cpath]
        tomtom_path = tomtom.loc[query_id, "geometry"]
        covered.append(tomtom_path.dwithin(points, 100).all())
    matches = matches.with_columns(pl.Series(covered).alias("covered"))
    n = matches["covered"].sum()
    print(f"Matches with covered path: {n}")
    return matches.filter(pl.col("covered")).drop("covered")


def save(tomtom: gpd.GeoDataFrame, matches: pl.DataFrame, output_file: str):
    print("Saving file...")
    matches = matches.join(
        pl.from_pandas(
            tomtom.loc[:, ["id", "departure_time", "tt_no_traffic", "tt_traffic", "tt_historic"]]
        ),
        on="id",
    )
    matches.select(
        "id",
        "cpath",
        "length",
        "length_tomtom",
        "departure_time",
        "tt_no_traffic",
        "tt_traffic",
        "tt_historic",
    )
    matches.write_parquet(output_file)


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "crs",
        "clean_edges_file",
        "calibration.tomtom.output_filename",
        "calibration.map_matching.output_filename",
        "calibration.post_map_matching.output_filename",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()

    edges = read_edges(config["clean_edges_file"], config["crs"])
    tomtom = read_tomtom(config["calibration"]["tomtom"]["output_filename"], config["crs"])
    matches = read_matches(config["calibration"]["map_matching"]["output_filename"])
    matches = clean_od(edges, tomtom, matches)
    matches = clean_paths(edges, matches)
    matches = clean_length(edges, tomtom, matches)
    matches = clean_covered(edges, tomtom, matches)
    save(tomtom, matches, config["calibration"]["post_map_matching"]["output_filename"])

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
