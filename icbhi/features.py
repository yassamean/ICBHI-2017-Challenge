"""Data integration and transformation: aggregate cycles and recordings to one row per patient."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CHILD_AGE_BANDS = [(0, 2, "0–1"), (2, 6, "2–5"), (6, 13, "6–12"), (13, 19, "13–18")]
ADULT_AGE_BANDS = [(19, 60, "19–59"), (60, 70, "60–69"), (70, 80, "70–79"), (80, 200, "80+")]


def aggregate_recordings(recordings: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Per-patient lung-sound features plus recording metadata (kept for colouring only)."""
    cycles = cycles.merge(recordings[["recording", "pid"]], on="recording")
    crackles, wheezes = cycles["crackles"] == 1, cycles["wheezes"] == 1
    # Mutually exclusive cycle types: together with normal cycles they add up to 100 %,
    # so no cycle is counted in more than one model input.
    cycles["crackles_only"] = crackles & ~wheezes
    cycles["wheezes_only"] = wheezes & ~crackles
    cycles["both"] = crackles & wheezes
    # Cut-off breaths at recording edges have no meaningful length.
    cycles["full_duration"] = cycles["duration"].where(~cycles["truncated"])

    sound = cycles.groupby("pid").agg(
        n_cycles=("cycle", "size"),
        crackle_rate=("crackles", "mean"),  # totals, incl. "both" cycles: for description only
        wheeze_rate=("wheezes", "mean"),
        crackle_only_rate=("crackles_only", "mean"),
        wheeze_only_rate=("wheezes_only", "mean"),
        both_rate=("both", "mean"),
        cycle_duration_mean=("full_duration", "mean"),
        cycle_duration_std=("full_duration", "std"),
    )

    meta = recordings.groupby("pid").agg(
        n_recordings=("recording", "size"),
        n_locations=("location", "nunique"),
        main_device=("device", lambda s: s.mode().iloc[0]),
        devices=("device", lambda s: " + ".join(sorted(s.unique()))),
        acquisition_mode=("mode", lambda s: " + ".join(sorted(s.unique()))),
        split=("split", "first"),
    )
    return sound.join(meta, how="outer").reset_index()


def age_band(age: float, is_child: bool) -> str:
    if pd.isna(age):
        return "Unknown"
    for low, high, label in CHILD_AGE_BANDS if is_child else ADULT_AGE_BANDS:
        if low <= age < high:
            return label
    return "Unknown"


def build_patient_table(
    patients: pd.DataFrame, recordings: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    table = patients.merge(aggregate_recordings(recordings, cycles), on="pid", how="left")
    table["age_band"] = [age_band(a, c) for a, c in zip(table["age"], table["is_child"])]
    table["sex_male"] = table["sex"].map({"M": 1.0, "F": 0.0})
    table["diagnosis_group"] = table["diagnosis"].map(config.DIAGNOSIS_GROUPS)
    return add_body_size_features(table)


def add_body_size_features(table: pd.DataFrame) -> pd.DataFrame:
    """Derive one body-size measure that exists for every patient.

    Child BMI is computed from weight and height. Because BMI is not comparable
    between children and adults (Rocha et al. 2019), the combined BMI is also
    z-scored within each age group; only the z-score is used across groups.
    Re-run this after imputing weight/height/BMI.
    """
    table = table.copy()
    table["child_bmi"] = table["child_weight"] / (table["child_height"] / 100) ** 2
    table["bmi_combined"] = np.where(table["is_child"], table["child_bmi"], table["adult_bmi"])
    grouped = table.groupby("age_group")["bmi_combined"]
    table["bmi_z_within_group"] = (table["bmi_combined"] - grouped.transform("mean")) / grouped.transform("std")
    return table
