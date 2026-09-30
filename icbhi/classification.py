"""Classification with repeated stratified cross-validation and imbalance handling.

Every step that learns from data - imputation, body-size z-scores, scaling and
resampling - sits inside the pipeline, so it is fitted on the training folds only.
Resampling is applied during fitting only; test folds keep their real class mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from . import missing
from .preprocess import SOUND_FEATURES

TARGETS = {
    "3 groups · all patients": {
        "subgroup": "All",
        "label": lambda df: df["diagnosis_group"],
        "body": ["bmi_z_within_group"],
    },
    "COPD vs other · adults": {
        "subgroup": "Adults",
        "label": lambda df: np.where(df["diagnosis"] == "COPD", "COPD", "Other"),
        "body": ["adult_bmi"],
    },
}
FEATURE_SETS = {
    "All features": "demographics + body size + lung sounds",
    "Lung sounds only": "crackle / wheeze rates and breathing-cycle durations",
    "Demographics only": "age, sex and body size",
}
MODELS = {
    "Baseline (always majority class)": lambda seed: DummyClassifier(strategy="most_frequent"),
    # Depth and leaf size are limited so the tree cannot memorise 126 patients.
    "Decision tree": lambda seed: DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=seed),
    "k-nearest neighbours": lambda seed: KNeighborsClassifier(n_neighbors=5),
    "Random forest": lambda seed: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ),
}
SAMPLERS = ["None", "Random undersampling", "SMOTE oversampling"]
SAMPLER_NOTES = {
    "None": "Train on the real class mix.",
    "Random undersampling": "Randomly drop majority-class patients from the training folds until all classes are equally large.",
    "SMOTE oversampling": "Create synthetic minority patients between existing ones (k nearest neighbours) until all "
    "classes are equally large. SMOTE-NC is used when sex is a feature, so no fractional sex values are created.",
}


def feature_columns(target: str, feature_set: str) -> list[str]:
    demographics = ["age", "sex_male", *TARGETS[target]["body"]]
    return {
        "All features": [*demographics, *SOUND_FEATURES],
        "Lung sounds only": list(SOUND_FEATURES),
        "Demographics only": demographics,
    }[feature_set]


BODY_COLUMNS = ["adult_bmi", "child_weight", "child_height"]


def impute_fold(data: pd.DataFrame, donors: pd.Index, method: str, seed: int = 0) -> pd.DataFrame:
    """Impute body-size gaps of all rows, using only the `donors` (training patients) to fit.

    Training and test rows are filled in one call, but only training rows provide
    observed values / fit the imputation model, so no test information leaks in.
    """
    columns = [c for c in BODY_COLUMNS if data[c].isna().any()]
    if not columns:
        return data
    return missing.impute(data, columns, method, random_state=seed, donors=donors, n_imputations=3).data


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Derive the BMI z-score with training statistics and select the model's input columns."""

    def __init__(self, columns: tuple[str, ...] = ()):
        self.columns = columns

    def _bmi(self, X: pd.DataFrame) -> pd.Series:
        child_bmi = X["child_weight"] / (X["child_height"] / 100) ** 2
        return pd.Series(np.where(X["is_child"], child_bmi, X["adult_bmi"]), index=X.index)

    def fit(self, X: pd.DataFrame, y=None):
        bmi = self._bmi(X)
        self.stats_ = bmi.groupby(X["is_child"]).agg(["mean", "std"])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        if "bmi_z_within_group" in self.columns:
            stats = self.stats_.reindex(X["is_child"])
            X["bmi_z_within_group"] = (self._bmi(X).to_numpy() - stats["mean"].to_numpy()) / stats["std"].to_numpy()
        return X[list(self.columns)].to_numpy(dtype=float)


def make_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    subgroup = TARGETS[target]["subgroup"]
    data = df if subgroup == "All" else df[~df["is_child"]]
    return data, np.asarray(TARGETS[target]["label"](data))


def build_pipeline(
    target: str, feature_set: str, model: str, sampler: str, y: np.ndarray, seed: int = 0, n_splits: int = 5
) -> Pipeline:
    """Feature derivation → scaling → resampling → classifier (imputation happens per fold before this)."""
    columns = feature_columns(target, feature_set)
    steps = [
        ("features", FeatureBuilder(tuple(columns))),
        ("scale", StandardScaler()),
    ]
    if sampler == "Random undersampling":
        steps.append(("resample", RandomUnderSampler(random_state=seed)))
    elif sampler == "SMOTE oversampling":
        # SMOTE needs k neighbours of the same class inside each training fold.
        smallest = int(pd.Series(y).value_counts().min() * (n_splits - 1) / n_splits)
        k = max(1, min(5, smallest - 1))
        if "sex_male" in columns:
            steps.append(("resample", SMOTENC([columns.index("sex_male")], k_neighbors=k, random_state=seed)))
        else:
            steps.append(("resample", SMOTE(k_neighbors=k, random_state=seed)))
    steps.append(("model", MODELS[model](seed)))
    return Pipeline(steps)


@dataclass
class Evaluation:
    folds: pd.DataFrame  # one row per (repeat, fold) with all metrics
    confusion: pd.DataFrame  # true × predicted, averaged per repeat
    classes: list[str]

    def summary(self) -> pd.Series:
        metrics = self.folds.drop(columns=["repeat", "fold"])
        return pd.concat({"mean": metrics.mean(), "std": metrics.std()}, axis=1).stack()


def evaluate_many(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
    models: list[str],
    samplers: list[str],
    imputation: str = "median",
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[tuple[str, str], Evaluation]:
    """Repeated stratified k-fold CV for every (model, sampler) on the same folds (paired comparison).

    Each fold is imputed once and reused for all models and samplers.
    """
    data, y = make_dataset(df, target)
    classes = sorted(np.unique(y))
    minority = pd.Series(y).value_counts().idxmin()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = [(train, test, impute_fold(data, data.index[train], imputation, seed)) for train, test in cv.split(data, y)]

    results = {}
    for model in models:
        for sampler in samplers:
            template = build_pipeline(target, feature_set, model, sampler, y, seed, n_splits)
            rows, matrix = [], np.zeros((len(classes), len(classes)))
            for i, (train, test, imputed) in enumerate(folds):
                pipeline = clone(template).fit(imputed.iloc[train], y[train])
                predicted = pipeline.predict(imputed.iloc[test])
                proba = pipeline.predict_proba(imputed.iloc[test])
                model_classes = list(pipeline.classes_)
                if len(classes) == 2:
                    auc = roc_auc_score(y[test] == minority, proba[:, model_classes.index(minority)])
                else:
                    auc = roc_auc_score(y[test], proba, multi_class="ovr", average="macro", labels=model_classes)
                rows.append({
                    "repeat": i // n_splits + 1,
                    "fold": i % n_splits + 1,
                    "F1 (macro)": f1_score(y[test], predicted, average="macro", zero_division=0),
                    f"F1 ({minority})": f1_score(y[test], predicted, labels=[minority], average="macro", zero_division=0),
                    "AUC": auc,
                    "accuracy": accuracy_score(y[test], predicted),
                })
                matrix += confusion_matrix(y[test], predicted, labels=classes)
            confusion = pd.DataFrame(matrix / n_repeats, index=classes, columns=classes)
            results[(model, sampler)] = Evaluation(pd.DataFrame(rows), confusion, classes)
    return results


def evaluate(
    df: pd.DataFrame,
    target: str,
    feature_set: str = "All features",
    model: str = "Random forest",
    sampler: str = "None",
    imputation: str = "median",
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> Evaluation:
    """Single (model, sampler) evaluation; the same seed gives the same folds, so runs are paired."""
    return evaluate_many(df, target, feature_set, [model], [sampler], imputation, n_splits, n_repeats, seed)[(model, sampler)]


def fitted_model(df: pd.DataFrame, target: str, feature_set: str, model: str, imputation: str, seed: int = 0):
    """Fit on all patients (for showing the tree's rules / feature importance, not for scoring)."""
    data, y = make_dataset(df, target)
    data = impute_fold(data, data.index, imputation, seed)
    pipeline = build_pipeline(target, feature_set, model, "None", y, seed).fit(data, y)
    return pipeline, feature_columns(target, feature_set)


def tree_rules(df: pd.DataFrame, target: str, feature_set: str, imputation: str, seed: int = 0) -> str:
    pipeline, columns = fitted_model(df, target, feature_set, "Decision tree", imputation, seed)
    # Report thresholds in original units: refit on unscaled features for readable rules.
    data, y = make_dataset(df, target)
    features = pipeline[:1].transform(impute_fold(data, data.index, imputation, seed))
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=seed).fit(features, y)
    return export_text(tree, feature_names=columns, decimals=2)


def feature_importance(df: pd.DataFrame, target: str, feature_set: str, imputation: str, seed: int = 0) -> pd.Series:
    pipeline, columns = fitted_model(df, target, feature_set, "Random forest", imputation, seed)
    return pd.Series(pipeline["model"].feature_importances_, index=columns).sort_values(ascending=False)
