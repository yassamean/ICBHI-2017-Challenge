"""Feature selection, subgroup selection and normalisation for dimensionality reduction."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .features import add_body_size_features

# Mutually exclusive shares of a patient's cycles (the remaining share is normal cycles).
SOUND_FEATURES = ["crackle_only_rate", "wheeze_only_rate", "both_rate", "cycle_duration_mean", "cycle_duration_std"]
DEMOGRAPHIC_FEATURES = ["age", "sex_male"]
BODY_FEATURES = {
    "All": ["bmi_z_within_group"],
    "Adults": ["adult_bmi"],
    "Children": ["child_weight", "child_height"],
}
SUBGROUPS = list(BODY_FEATURES)

SCALERS = {
    "Standard (z-score)": StandardScaler,
    "Robust (median / IQR)": RobustScaler,
    "Min–max [0, 1]": MinMaxScaler,
}

# What each attribute is used for and why; shown in the app.
FEATURE_ROLES = pd.DataFrame(
    [
        ("age", "Input", "Demographic"),
        ("sex", "Input (as sex_male 0/1)", "Demographic"),
        ("adult_bmi", "Input (adults view)", "Body size; only defined for adults"),
        ("child_weight, child_height", "Input (children view)", "Body size; only defined for children"),
        ("bmi_z_within_group", "Input (all-patients view)",
         "BMI z-scored within adults / children, since raw child and adult BMI are not comparable"),
        ("crackle_only_rate, wheeze_only_rate, both_rate", "Input",
         "Share of the patient's breathing cycles with only crackles, only wheezes, or both; mutually exclusive, "
         "the rest are normal cycles"),
        ("crackle_rate, wheeze_rate", "Description only",
         "Total shares including the 'both' cycles; as inputs next to both_rate they would count those cycles twice"),
        ("cycle_duration_mean, cycle_duration_std", "Input", "Breathing pattern (length and regularity of cycles)"),
        ("diagnosis", "Colour only", "The label; as an input the embedding would just reproduce it"),
        ("main_device, devices, acquisition_mode", "Colour only",
         "How the patient was recorded, not the patient; confounded with study and age group"),
        ("split", "Colour only", "Official train/test assignment"),
        ("n_recordings, n_cycles", "Dropped", "Depend on the study protocol (multichannel studies produce more files)"),
        ("pid, recording index, file name", "Dropped", "Identifiers"),
    ],
    columns=["attribute", "role", "reason"],
)

META_COLUMNS = [
    "pid", "age", "sex", "age_group", "age_band", "diagnosis", "diagnosis_group", "main_device",
    "devices", "acquisition_mode", "split", "n_recordings",
]


def select_subgroup(df: pd.DataFrame, subgroup: str = "All", sex: str | None = None) -> pd.DataFrame:
    if subgroup == "Adults":
        df = df[~df["is_child"]]
    elif subgroup == "Children":
        df = df[df["is_child"]]
    if sex in ("M", "F"):
        df = df[df["sex"] == sex]
    return df


def input_features(subgroup: str) -> list[str]:
    return [*DEMOGRAPHIC_FEATURES, *BODY_FEATURES[subgroup], *SOUND_FEATURES]


def build_feature_matrix(
    df: pd.DataFrame,
    subgroup: str = "All",
    sex: str | None = None,
    scaler: str = "Standard (z-score)",
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Return (scaled X, metadata for colouring, pids dropped for remaining missing values).

    Expects imputation to have happened already; any patient still missing an input
    feature is dropped (listwise) and reported. Constant columns (e.g. sex after a
    sex filter) are removed because they carry no information and break scaling.
    """
    data = select_subgroup(add_body_size_features(df), subgroup, sex)
    features = input_features(subgroup)
    incomplete = data[features].isna().any(axis=1)
    dropped = data.loc[incomplete, "pid"].astype(int).tolist()
    data = data[~incomplete]

    X = data[features]
    X = X.loc[:, X.nunique() > 1]
    scaled = pd.DataFrame(SCALERS[scaler]().fit_transform(X), index=X.index, columns=X.columns)
    meta = data[[c for c in META_COLUMNS if c in data.columns]]
    return scaled, meta, dropped
