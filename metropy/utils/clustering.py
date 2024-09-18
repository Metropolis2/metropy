import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def _get_variables(lf: pl.LazyFrame):
    variables = [
        "od_distance",
        "preceding_purpose",
        "following_purpose",
        "area_origin",
        "area_destination",
        "departement_origin",
        "departement_destination",
    ]
    # Normalize the variables so that:
    # - A difference in purpose (preceding or following) is equivalent to 1.
    # - A difference in area (origin or destination) is equivalent to 1 or 2.
    # - A difference in departement (origin or destination) is equivalent to 0.5.
    # - Differences in sqrt(od_distance) range from 0 to 10.
    x = (
        lf.select(variables)
        .collect()
        .to_dummies(~(cs.by_name("od_distance") | cs.starts_with("area_")))
        .with_columns(
            cs.starts_with("departement_") * np.sqrt(0.5),
            od_distance=np.sqrt(10)
            * pl.col("od_distance").sqrt()
            / pl.col("od_distance").max().sqrt(),
        )
        .to_pandas()
    )
    return x


def trip_clustering(lf: pl.LazyFrame, cluster_size=1000, random_seed=None):
    print("Running clustering...")
    x = _get_variables(lf)
    nb_clusters = len(x) // cluster_size
    clustering = KMeans(nb_clusters, random_state=random_seed).fit(x)
    centers = pl.DataFrame(clustering.cluster_centers_, schema=list(x.columns)).with_columns(
        cluster=pl.int_range(nb_clusters, eager=True)
    )
    assert clustering.labels_ is not None
    print("Number of clusters: {:,}".format(nb_clusters))
    print("Number of trips classified: {:,}".format(len(clustering.labels_)))
    return clustering.labels_, centers


def trip_labeling(lf: pl.LazyFrame, centers: pl.DataFrame):
    print("Assigning cluster to trips...")
    assert centers["cluster"].min() == 0
    assert centers["cluster"].max() == len(centers) - 1
    x = _get_variables(lf)
    y = centers.sort("cluster").drop("cluster").to_pandas()
    labels = pairwise_distances_argmin(x, y, metric="euclidean")
    return labels
