"""Tests for clustering, dimensionality reduction and the classification pipeline pieces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from icbhi import classification, clustering, reduction


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.3, (20, 3)), rng.normal(5, 0.3, (15, 3)), rng.normal([0, 8, 0], 0.3, (10, 3))])
    return pd.DataFrame(points, columns=["a", "b", "c"]), np.repeat([0, 1, 2], [20, 15, 10])


def _same_partition(a, b):
    return len(pd.crosstab(a, b).stack().loc[lambda s: s > 0]) == len(np.unique(a)) == len(np.unique(b))


def test_diana_recovers_separated_groups(blobs):
    X, truth = blobs
    labels = clustering.diana(squareform(pdist(X.to_numpy())), 3)
    assert _same_partition(labels, truth)


@pytest.mark.parametrize("method", ["k-means", "AGNES", "DIANA"])
def test_partitioning_and_hierarchical_methods(blobs, method):
    X, truth = blobs
    labels = clustering.cluster(X, method, n_clusters=3)
    assert _same_partition(labels, truth)
    assert clustering.quality(X, labels)["silhouette"] > 0.8


def test_dbscan_marks_outlier_as_noise(blobs):
    X, _ = blobs
    X = pd.concat([X, pd.DataFrame([[20.0, 20.0, 20.0]], columns=X.columns)], ignore_index=True)
    labels = clustering.cluster(X, "DBSCAN", eps=1.0, min_samples=4)
    assert labels[-1] == clustering.NOISE
    quality = clustering.quality(X, labels)
    assert quality["clusters"] == 3 and quality["noise share"] == pytest.approx(1 / len(X))


def test_quality_intra_smaller_than_inter(blobs):
    X, truth = blobs
    q = clustering.quality(X, truth)
    assert q["mean intra-cluster distance"] < q["mean inter-cluster distance"]


def test_align_labels_undoes_renumbering():
    reference = np.array([0, 0, 1, 1, 2, 2, -1])
    renumbered = np.array([2, 2, 0, 0, 1, 1, -1])
    assert (clustering.align_labels(reference, renumbered) == reference).all()


def test_composition_dominant_share():
    table = clustering.composition(np.array([0, 0, 0, 1]), pd.Series(["a", "a", "b", "b"]))
    assert table.loc["Cluster 1", "dominant group"] == "a"
    assert table.loc["Cluster 1", "dominant share"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("method", reduction.METHODS)
def test_embeddings_have_right_shape(blobs, method):
    X, _ = blobs
    embedding = reduction.embed(X, method)
    assert embedding.shape == (len(X), 2)
    assert 0 <= reduction.neighbour_preservation(X, embedding) <= 1


def _patients(child_heights):
    n = len(child_heights)
    return pd.DataFrame(
        {
            "age": np.linspace(1, 15, n),
            "sex_male": np.tile([0.0, 1.0], n)[:n],
            "is_child": True,
            "adult_bmi": np.nan,
            "child_weight": np.linspace(10, 50, n),
            "child_height": child_heights,
        }
    )


def test_fold_imputation_uses_training_donors_only():
    data = _patients([80.0, 90.0, 100.0, 110.0, np.nan, 500.0])  # rows 4-5 are the test fold
    imputed = classification.impute_fold(data, data.index[:4], "median")
    assert imputed.loc[4, "child_height"] == 95.0  # the test value 500 must not influence the median


@pytest.mark.parametrize("method", ["knn", "mice", "regression"])
def test_fold_imputation_fills_test_rows(method):
    data = _patients([80.0, 90.0, 100.0, 110.0, 120.0, np.nan])
    imputed = classification.impute_fold(data, data.index[:5], method)
    assert imputed["child_height"].notna().all()


def test_feature_builder_uses_training_statistics():
    train = pd.DataFrame(
        {"is_child": [False, False, True, True], "adult_bmi": [20.0, 30.0, np.nan, np.nan],
         "child_weight": [np.nan, np.nan, 16.0, 25.0], "child_height": [np.nan, np.nan, 100.0, 100.0]}
    )
    builder = classification.FeatureBuilder(("bmi_z_within_group",)).fit(train)
    out = builder.transform(pd.DataFrame(
        {"is_child": [False], "adult_bmi": [25.0], "child_weight": [np.nan], "child_height": [np.nan]}
    ))
    assert out[0, 0] == pytest.approx(0.0)  # 25 is the training mean of adult BMI
