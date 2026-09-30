"""Dimensionality reduction (PCA, t-SNE, UMAP, PaCMAP) and simple structure-preservation checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

METHODS = ["PCA", "t-SNE", "UMAP", "PaCMAP"]

# Short descriptions shown in the app, following the Towards Data Science post.
DESCRIPTIONS = {
    "PCA": "Linear projection onto the directions of largest variance. Keeps global structure, "
    "misses non-linear patterns. Used as a reference.",
    "t-SNE": "Keeps close neighbours close; distances between clusters and cluster sizes are not "
    "meaningful. Sensitive to perplexity.",
    "UMAP": "Graph of nearest neighbours; faster than t-SNE and keeps somewhat more global "
    "structure, but still mainly local. Sensitive to n_neighbors.",
    "PaCMAP": "Uses near, mid-near and far pairs to preserve local and global structure with "
    "good defaults. Recommended first choice in the blog post.",
}


def embed(
    X: pd.DataFrame,
    method: str,
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
) -> pd.DataFrame:
    """Embed the rows of a scaled feature matrix into `n_components` dimensions."""
    data = X.to_numpy(dtype=float)
    n = len(data)
    if method == "PCA":
        result = PCA(n_components=n_components, random_state=random_state).fit_transform(data)
    elif method == "t-SNE":
        result = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, (n - 1) / 3),
            init="pca",
            random_state=random_state,
        ).fit_transform(data)
    elif method == "UMAP":
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # n_jobs is forced to 1 when random_state is set
            result = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, n - 1),
                random_state=random_state,
            ).fit_transform(data)
    elif method == "PaCMAP":
        import pacmap

        result = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, n - 2) if n_neighbors else None,
            random_state=random_state,
        ).fit_transform(data, init="pca")
    else:
        raise ValueError(f"Unknown method {method!r}")
    columns = [f"dim {i + 1}" for i in range(n_components)]
    return pd.DataFrame(result, index=X.index, columns=columns)


def neighbour_preservation(X: pd.DataFrame, embedding: pd.DataFrame, k: int = 10) -> float:
    """Local structure: average share of each point's k nearest neighbours kept in the embedding."""
    k = min(k, len(X) - 1)

    def neighbours(data):
        index = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data, return_distance=False)
        return [set(row[1:]) for row in index]

    original, embedded = neighbours(X.to_numpy()), neighbours(embedding.to_numpy())
    return float(np.mean([len(a & b) / k for a, b in zip(original, embedded)]))


def distance_correlation(X: pd.DataFrame, embedding: pd.DataFrame) -> float:
    """Global structure: rank correlation between all pairwise distances before and after."""
    return float(spearmanr(pdist(X.to_numpy()), pdist(embedding.to_numpy())).statistic)
