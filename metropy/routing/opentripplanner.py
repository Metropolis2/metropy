import os
import time
import re
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor

import polars as pl

import metropy.utils.io as metro_io

NB_TRIES = 3

BODY = """
query GetPlan(
    $date: String,
    $time: String,
    $from: InputCoordinates,
    $to: InputCoordinates,
    $numItineraries: Int
    $searchWindow: Long,
    $walkReluctance: Float
    $waitReluctance: Float
    $walkSpeed: Float
    $arriveBy: Boolean
    $walkBoardCost: Int
    $transferPenalty: Int
    $transportModes: [TransportMode]
    $modeWeight: InputModeWeight
    $minTransferTime: Int
    $ignoreRealtimeUpdates: Boolean
) {
    plan(
        date: $date
        time: $time
        from: $from
        to: $to
        numItineraries: $numItineraries
        searchWindow: $searchWindow
        walkReluctance: $walkReluctance
        waitReluctance: $waitReluctance
        walkSpeed: $walkSpeed
        arriveBy: $arriveBy
        walkBoardCost: $walkBoardCost
        transferPenalty: $transferPenalty
        transportModes: $transportModes
        modeWeight: $modeWeight
        minTransferTime: $minTransferTime
        ignoreRealtimeUpdates: $ignoreRealtimeUpdates
        debugItineraryFilter: true
    ) {
        itineraries {
            generalizedCost
            duration
            systemNotices {
                tag
                text
            }
            legs {
                mode
                duration
                transitLeg
                from {
                    stop {
                        gtfsId
                    }
                }
                to {
                    stop {
                        gtfsId
                    }
                }
                route {
                    gtfsId
                }
            }
        }
    }
}
"""

HEADERS = {"Content-Type": "application/json", "OTPTimeout": "180000"}


def read_origin_destination_pairs(input_directory: str, config: dict):
    print("Reading trips' coordinates...")
    columns = [
        "trip_id",
        "origin_lng",
        "origin_lat",
        "destination_lng",
        "destination_lat",
    ]
    if config["time"] == "departure":
        columns.append("departure_time")
    elif config["time"] == "arrival":
        columns.append("arrival_time")
    elif config["time"] == "tstar":
        columns.append("tstar")
    df = pl.read_parquet(
        os.path.join(input_directory, "trips.parquet"),
        columns=columns,
    )
    df = df.filter(
        (pl.col("origin_lng") != pl.col("destination_lng"))
        | (pl.col("origin_lat") != pl.col("destination_lat"))
    )
    if config["time"] in ("departure", "arrival", "tstar"):
        df = df.rename({columns[-1]: "time"}).with_columns(pl.col("time").cast(pl.UInt64))
        df = df.with_columns(
            pl.format(
                "{}:{}:{}",
                (pl.col("time") // 3600 % 24).cast(pl.String).str.pad_start(2, "0"),
                (pl.col("time") % 3600 // 60).cast(pl.String).str.pad_start(2, "0"),
                (pl.col("time") % 60).cast(pl.String).str.pad_start(2, "0"),
            ).alias("time"),
        )
    else:
        assert re.fullmatch(
            "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]", config["time"]
        ), f"Invalid key `routing.opentripplanner.time`: {config['time']}"
        df = df.with_columns(pl.lit(config["time"]).alias("time"))
    df = df.with_columns(pl.lit(config["time"] in ("arrival", "tstar")).alias("arrive_by"))
    return df


def clean_leg(leg: dict, leg_id: int):
    if leg["transitLeg"]:
        route_id = leg["route"]["gtfsId"]
        from_stop = leg["from"]["stop"]["gtfsId"]
        to_stop = leg["to"]["stop"]["gtfsId"]
    else:
        route_id = None
        from_stop = None
        to_stop = None
    clean_leg = {
        "leg_index": leg_id,
        "mode": leg["mode"],
        "travel_time": leg["duration"],
        "route_id": route_id,
        "from_stop_id": from_stop,
        "to_stop_id": to_stop,
    }
    return clean_leg


def get_least_cost_itinerary(row, config: dict, nb_tries=0):
    variables = {
        "date": config["date"],
        "time": row["time"],
        "from": {
            "lat": row["origin_lat"],
            "lon": row["origin_lng"],
        },
        "to": {
            "lat": row["destination_lat"],
            "lon": row["destination_lng"],
        },
        "numItineraries": 6,
        "searchWindow": 1800,
        "walkReluctance": 2.0,
        "waitReluctance": 1.1,
        "walkSpeed": 4.0 / 3.6,
        "arriveBy": row["arrive_by"],
        "walkBoardCost": 600,
        "transferPenalty": 0,
        "transportModes": [
            {"mode": "WALK"},
            {"mode": "TRANSIT"},
        ],
        "modeWeight": {
            "TRAM": 1.0,
            "RAIL": 1.0,
            "BUS": 1.2,
            "SUBWAY": 1.0,
        },
        "minTransferTime": 0,
        "ignoreRealtimeUpdates": True,
    }
    req = requests.post(
        config["url"],
        headers=HEADERS,
        json={
            "query": BODY,
            "variables": variables,
        },
    )
    data = req.json()
    if "errors" in data:
        if nb_tries > NB_TRIES:
            print(data)
            return (row["trip_id"], None, None, None)
        else:
            # Retry.
            return get_least_cost_itinerary(row, config, nb_tries + 1)
    if not "plan" in data["data"].keys():
        raise Exception(data)
    # Find the itinerary with the least cost.
    it = min(
        filter(lambda it: len(it["legs"]) > 0, data["data"]["plan"]["itineraries"]),
        key=lambda it: it["generalizedCost"],
        default=None,
    )
    if it is None:
        return (row["trip_id"], None, None, None)
    legs = [clean_leg(leg, i) for i, leg in enumerate(it["legs"])]
    return row["trip_id"], it["duration"], it["generalizedCost"], legs


def run_queries_batch(ods: pl.DataFrame, config: dict):
    print("Running queries")
    with ThreadPoolExecutor(max_workers=config.get("nb_threads", 12)) as executor:
        futures = [
            executor.submit(get_least_cost_itinerary, row, config)
            for row in ods.iter_rows(named=True)
        ]
    results = [future.result() for future in futures]
    df = pl.from_records(
        results,
        schema=[
            ("trip_id", pl.UInt64),
            ("travel_time", pl.Float64),
            ("generalized_cost", pl.Float64),
            (
                "legs",
                pl.List(
                    pl.Struct(
                        [
                            pl.Field("leg_index", pl.UInt8),
                            pl.Field("mode", pl.String),
                            pl.Field("travel_time", pl.Float64),
                            pl.Field("route_id", pl.String),
                            pl.Field("from_stop_id", pl.String),
                            pl.Field("to_stop_id", pl.String),
                        ]
                    )
                ),
            ),
        ],
    )
    df = df.with_columns(
        (
            (pl.col("legs").list.len() == 1)
            & (pl.col("legs").list.first().struct.field("mode") == "WALK")
        ).alias("walk_only")
    )
    df = df.drop_nulls()
    return df


def run_queries(ods: pl.DataFrame, config: dict, tmp_directory: str):
    batch_size = config.get("batch_size", len(ods))
    if batch_size == 0:
        batch_size = len(ods)
    nb_batches = len(ods) // batch_size + (len(ods) % batch_size != 1)
    if nb_batches == 1:
        return run_queries_batch(ods, config)
    for i in range(nb_batches):
        df = run_queries_batch(ods[i * batch_size : (i + 1) * batch_size], config)
        df.write_parquet(os.path.join(tmp_directory, f"otp_results_{i}.parquet"))
        del df
    df = pl.concat(
        (
            pl.scan_parquet(os.path.join(tmp_directory, f"otp_results_{i}.parquet"))
            for i in range(nb_batches)
        ),
        how="vertical",
    ).collect()
    return df


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "tmp_directory",
        "routing.opentripplanner.url",
        "routing.opentripplanner.output_filename",
        "routing.opentripplanner.date",
        "routing.opentripplanner.time",
    ]
    check_keys(config, mandatory_keys)
    otp_config = config["routing"]["opentripplanner"]

    if not os.path.isdir(config["tmp_directory"]):
        os.makedirs(config["tmp_directory"])

    t0 = time.time()

    ods = read_origin_destination_pairs(config["population_directory"], otp_config)

    df = run_queries(ods, otp_config, config["tmp_directory"])

    metro_io.save_dataframe(
        df,
        otp_config["output_filename"],
    )

    # Delete temporary directory.
    try:
        shutil.rmtree(config["tmp_directory"])
    except OSError as e:
        print(e)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
