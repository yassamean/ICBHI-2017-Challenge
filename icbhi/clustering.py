"""Clustering (partitioning, hierarchical, density-based) and cluster quality measures.

Quality follows the lecture's definition - high intra-cluster similarity, low
inter-cluster similarity, both depending on the chosen distance function - plus the
Silhouette coefficient named in the task sheet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

METHODS = {
    "k-means": "Partitioning",
    "AGNES": "Hierarchical (agglomerative, bottom-up)",
    "DIANA": "Hierarchical (divisive, top-down)",
    "DBSCAN": "Density-based",
    "OPTICS": "Density-based",
}
METRICS = {"Euclidean": "euclidean", "Manhattan": "cityblock"}
LINKAGES = ["ward", "average", "complete", "single"]
NOISE = -1


def cluster(
    X: pd.DataFrame,
    method: str,
    n_clusters: int = 3,
    metric: str = "Euclidean",
    linkage: str = "average",
    eps: float = 1.5,
    min_samples: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Return one cluster label per row; -1 marks noise for DBSCAN/OPTICS."""
    data = X.to_numpy(dtype=float)
    distance = METRICS[metric]
    if method == "k-means":  # k-means is defined for Euclidean distance only
        return KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(data)
    if method == "AGNES":
        if linkage == "ward":  # Ward minimises variance, so it requires Euclidean distance
            distance = "euclidean"
        return AgglomerativeClustering(n_clusters=n_clusters, metric=distance, linkage=linkage).fit_predict(data)
    if method == "DIANA":
        return diana(squareform(pdist(data, metric=distance)), n_clusters)
    if method == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples, metric=distance).fit_predict(data)
    if method == "OPTICS":
        return OPTICS(min_samples=min_samples, metric=distance).fit_predict(data)
    raise ValueError(f"Unknown method {method!r}")


def diana(distances: np.ndarray, n_clusters: int) -> np.ndarray:
    """DIANA (DIvisive ANAlysis, Kaufman & Rousseeuw 1990) cut at `n_clusters`.

    Start with one cluster. Repeatedly take the cluster with the largest diameter
    (largest pairwise distance) and split it: the point with the highest average
    distance to the others starts a "splinter group"; points then move to the
    splinter group one at a time while they are, on average, closer to it than to
    the rest.
    """
    n = len(distances)
    labels = np.zeros(n, dtype=int)
    for new_label in range(1, min(n_clusters, n)):
        diameters = {
            c: distances[np.ix_(labels == c, labels == c)].max() for c in np.unique(labels)
        }
        target = max(diameters, key=diameters.get)
        members = list(np.flatnonzero(labels == target))
        if len(members) < 2:
            break

        within = distances[np.ix_(members, members)]
        first = members[int(np.argmax(within.sum(axis=1) / (len(members) - 1)))]
        splinter, rest = [first], [m for m in members if m != first]
        while len(rest) > 1:
            to_rest = np.array([distances[i, [r for r in rest if r != i]].mean() for i in rest])
            to_splinter = np.array([distances[i, splinter].mean() for i in rest])
            gain = to_rest - to_splinter
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            splinter.append(rest.pop(best))
        labels[splinter] = new_label
    return labels


# --- Quality ------------------------------------------------------------------------


def quality(X: pd.DataFrame, labels: np.ndarray, metric: str = "Euclidean") -> dict:
    """Silhouette, mean intra- and inter-cluster distance and noise share.

    Noise points (-1) are excluded from all distance-based measures.
    """
    data = X.to_numpy(dtype=float)
    labels = np.asarray(labels)
    clustered = labels != NOISE
    found = np.unique(labels[clustered])
    result = {
        "clusters": len(found),
        "noise share": float((~clustered).mean()),
        "silhouette": np.nan,
        "mean intra-cluster distance": np.nan,
        "mean inter-cluster distance": np.nan,
    }
    if len(found) < 2 or clustered.sum() <= len(found):
        return result

    distances = squareform(pdist(data[clustered], metric=METRICS[metric]))
    kept = labels[clustered]
    same = kept[:, None] == kept[None, :]
    off_diagonal = ~np.eye(len(kept), dtype=bool)
    result["silhouette"] = float(silhouette_score(distances, kept, metric="precomputed"))
    result["mean intra-cluster distance"] = float(distances[same & off_diagonal].mean())
    result["mean inter-cluster distance"] = float(distances[~same].mean())
    return result


def silhouette_per_point(X: pd.DataFrame, labels: np.ndarray, metric: str = "Euclidean") -> np.ndarray:
    labels = np.asarray(labels)
    values = np.full(len(labels), np.nan)
    clustered = labels != NOISE
    if len(np.unique(labels[clustered])) >= 2:
        values[clustered] = silhouette_samples(
            X.to_numpy(dtype=float)[clustered], labels[clustered], metric=METRICS[metric]
        )
    return values


def within_cluster_sse(X: pd.DataFrame, labels: np.ndarray) -> float:
    """Sum of squared distances to the cluster mean (the quantity k-means minimises)."""
    data = X.to_numpy(dtype=float)
    return float(
        sum(((data[labels == c] - data[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels) if c != NOISE)
    )


def k_sweep(X: pd.DataFrame, method: str, ks=range(2, 9), **kwargs) -> pd.DataFrame:
    """Quality for a range of cluster counts (elbow and Silhouette curves)."""
    rows = []
    for k in ks:
        labels = cluster(X, method, n_clusters=k, **kwargs)
        rows.append({"k": k, **quality(X, labels, kwargs.get("metric", "Euclidean")),
                     "within-cluster SSE": within_cluster_sse(X, labels)})
    return pd.DataFrame(rows)


def k_distance(X: pd.DataFrame, k: int, metric: str = "Euclidean") -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour (for choosing DBSCAN's eps)."""
    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric=METRICS[metric]).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


# --- Interpretation -----------------------------------------------------------------


def cluster_names(labels: np.ndarray) -> np.ndarray:
    return np.array(["Noise" if label == NOISE else f"Cluster {label + 1}" for label in labels])


def composition(labels: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Cross-table of cluster × group, plus the dominant group and its share per cluster."""
    table = pd.crosstab(pd.Series(cluster_names(labels), index=groups.index, name="cluster"), groups)
    table["size"] = table.sum(axis=1)
    counts = table.drop(columns="size")
    table["dominant group"] = counts.idxmax(axis=1)
    table["dominant share"] = counts.max(axis=1) / table["size"]
    return table


def align_labels(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Renumber `labels` to match `reference` as closely as possible (cluster numbers are arbitrary)."""
    ref_ids = [c for c in np.unique(reference) if c != NOISE]
    new_ids = [c for c in np.unique(labels) if c != NOISE]
    overlap = np.array([[np.sum((labels == n) & (reference == r)) for r in ref_ids] for n in new_ids])
    rows, cols = linear_sum_assignment(-overlap) if overlap.size else ([], [])
    mapping = {new_ids[r]: ref_ids[c] for r, c in zip(rows, cols)}
    spare = iter(range(max(ref_ids + new_ids + [0]) + 1, 10_000))
    return np.array([label if label == NOISE else mapping.get(label) if label in mapping else next(spare) for label in labels])


def centroid_distances(X: pd.DataFrame, labels: np.ndarray, metric: str = "Euclidean") -> pd.DataFrame:
    names = cluster_names(labels)
    centroids = X.groupby(names).mean().drop(index="Noise", errors="ignore")
    return pd.DataFrame(
        cdist(centroids, centroids, metric=METRICS[metric]), index=centroids.index, columns=centroids.index
    )
