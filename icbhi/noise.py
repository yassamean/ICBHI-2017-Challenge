"""Noise identification: flag unusual values for review instead of silently removing them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


def iqr_outliers(df: pd.DataFrame, columns: list[str], k: float = 1.5, by: str | None = None) -> pd.DataFrame:
    """Values outside [Q1 - k·IQR, Q3 + k·IQR], optionally computed within groups of `by`."""
    rows = []
    groups = {"all": df} if by is None else dict(tuple(df.groupby(by)))
    for group, part in groups.items():
        for column in columns:
            values = part[column].dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
            for pid, value in part.loc[values.index[(values < low) | (values > high)], ["pid", column]].to_numpy():
                rows.append(
                    {"pid": int(pid), "group": group, "feature": column, "value": value,
                     "lower fence": low, "upper fence": high}
                )
    return pd.DataFrame(rows, columns=["pid", "group", "feature", "value", "lower fence", "upper fence"])


def growth_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Children whose weight or height is implausible *for their age*.

    A plain IQR check is useless here because height depends strongly on age; instead
    each value is compared with a log(age) regression and flagged if its standardised
    residual exceeds `threshold`.
    """
    children = df[df["is_child"]]
    rows = []
    for column in ["child_weight", "child_height"]:
        known = children[children[column].notna()]
        x = np.log1p(known[["age"]].to_numpy())
        y = np.log(known[column].to_numpy())
        residuals = y - LinearRegression().fit(x, y).predict(x)
        z = residuals / residuals.std(ddof=2)
        for (_, row), score in zip(known.iterrows(), z):
            if abs(score) > threshold:
                rows.append({"pid": row["pid"], "age": row["age"], "feature": column,
                             "value": row[column], "z (for age)": round(score, 2)})
    return pd.DataFrame(rows, columns=["pid", "age", "feature", "value", "z (for age)"])


def isolation_forest_scores(X: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """Multivariate anomaly score (higher = more unusual) on an already scaled feature matrix."""
    forest = IsolationForest(n_estimators=500, random_state=random_state).fit(X)
    return pd.Series(-forest.score_samples(X), index=X.index, name="anomaly score")
