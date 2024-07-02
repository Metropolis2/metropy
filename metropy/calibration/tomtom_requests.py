from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import asyncio
import aiohttp
from tqdm import tqdm

import metropy.utils.io as metro_io

BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute/"
PARAMS = {"computeTravelTimeFor": "all", "traffic": True}


def read_edges(input_file: str):
    print("Reading edges...")
    gdf = metro_io.read_geodataframe(input_file)
    return gdf


def generate_random_nodes(edges: gpd.GeoDataFrame, random_seed, config: dict):
    print("Generating random origin-destination pairs...")
    rng = np.random.default_rng(random_seed)
    if "excluded_road_types" in config:
        mask = ~edges.loc[:, "road_type"].isin(config["excluded_road_types"])
        edges = gpd.GeoDataFrame(edges.loc[mask].copy())
    all_nodes = list(edges["source"])
    if len(all_nodes) < config["nb_waypoints"]:
        raise Exception("Not enough nodes to select origin-destination pairs from")
    selected_nodes = rng.choice(all_nodes, size=(config["nb_routes"], config["nb_waypoints"]))
    source_to_xy = (
        edges.set_index("source")["geometry"].apply(lambda geom: geom.coords[0]).to_dict()
    )
    coordinates = np.array([source_to_xy[node] for node in selected_nodes.flatten()])
    coordinates = coordinates.reshape(config["nb_routes"], config["nb_waypoints"], 2)
    # Switch latitude and longitude.
    coordinates = coordinates[:, :, ::-1]
    return selected_nodes, coordinates


async def get_tomtom_request(url, session, params):
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(response.status)
                print(response.reason)
                text = await response.text()
                print(text)
    except Exception as e:
        print(e)
        pass


async def process_batch(api_key, nodes, coordinates, batch_id: int, batch_size: int, params: dict):
    batch_results = []
    start_index = batch_id * batch_size
    end_index = min((batch_id + 1) * batch_size, len(nodes))
    nodes_batch = nodes[start_index:end_index]
    coordinates_batch = coordinates[start_index:end_index]
    with tqdm(total=len(nodes_batch), desc="Processing") as pbar:
        async with aiohttp.ClientSession() as session:
            for node_ids, points in zip(nodes_batch, coordinates_batch):
                waypoint_coords = ":".join([f"{lat},{lon}" for lat, lon in points])
                url = f"{BASE_URL}{waypoint_coords}/json?key={api_key}"
                data = await get_tomtom_request(url, session, params)
                if data and "routes" in data:
                    assert len(data["routes"]) == 1
                    route = data["routes"][0]
                    assert len(route["legs"]) == len(node_ids) - 1
                    for i, leg in enumerate(route["legs"]):
                        geom = LineString([[p["longitude"], p["latitude"]] for p in leg["points"]])
                        departure_time = datetime.fromisoformat(leg["summary"]["departureTime"])
                        res = {
                            "source": node_ids[i],
                            "target": node_ids[i + 1],
                            "length": leg["summary"]["lengthInMeters"],
                            "departure_time": departure_time,
                            "tt_no_traffic": leg["summary"]["noTrafficTravelTimeInSeconds"],
                            "tt_traffic": leg["summary"]["travelTimeInSeconds"],
                            "tt_historical": leg["summary"]["historicTrafficTravelTimeInSeconds"],
                            "geometry": geom,
                        }
                        batch_results.append(res)
                else:
                    pass
                pbar.update(1)
    gdf = gpd.GeoDataFrame(batch_results, crs="EPSG:4326")
    return gdf


async def get_tomtom_data(config, api_key, nodes, coordinates):
    print("Processing batches...")
    batch_size = int(np.ceil(config["nb_routes"] / config.get("nb_batches", 1)))
    params = PARAMS
    params["departAt"] = config["departure_time"]
    results = await asyncio.gather(
        *(
            process_batch(
                api_key,
                nodes,
                coordinates,
                i,
                batch_size,
                params,
            )
            for i in range(config.get("nb_batches", 1))
        )
    )
    gdf = gpd.GeoDataFrame(pd.concat(results), crs="EPSG:4326")
    gdf["id"] = np.arange(len(gdf))
    return gdf


if __name__ == "__main__":
    from metropy.config import read_config, check_keys, read_secrets

    config = read_config()
    mandatory_keys = [
        "clean_edges_file",
        "calibration.tomtom.output_file",
        "calibration.tomtom.nb_routes",
        "calibration.tomtom.nb_waypoints",
    ]
    check_keys(config, mandatory_keys)
    tomtom_config = config["calibration"]["tomtom"]

    secrets = read_secrets()
    if not "tomtom_key" in secrets:
        raise Exception("Missing key `tomtom_key` in secrets")

    edges = read_edges(config["clean_edges_file"])

    nodes, coordinates = generate_random_nodes(edges, config.get("random_seed"), tomtom_config)

    gdf = asyncio.run(get_tomtom_data(tomtom_config, secrets["tomtom_key"], nodes, coordinates))

    metro_io.save_geodataframe(gdf, tomtom_config["output_file"])
