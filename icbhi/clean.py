"""Data cleaning: consistency, validity and plausibility checks on the raw tables.

Every check appends an entry to a CleaningLog describing what was found and what
was done, so the app can show the cleaning steps transparently. Values are only
removed when they are logically impossible; merely unusual values are flagged and
left to the noise-identification step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Plausibility ranges (flag only). Wide on purpose: they catch typos and unit errors,
# not unusual-but-real patients.
PLAUSIBLE_RANGES = {
    "age": (0, 110),
    "adult_bmi": (12, 70),
    "child_weight": (1, 150),
    "child_height": (40, 210),
}
# No real breath is this short (it would mean >120 breaths/min, above even infant rates).
# Such cycles sit at the start/end of recordings, i.e. breaths cut off by the recording edge.
MIN_BREATH_SECONDS = 0.5


class CleaningLog:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, stage: str, check: str, finding: str, action: str = "None needed") -> None:
        self.entries.append({"stage": stage, "check": check, "finding": finding, "action": action})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["stage", "check", "finding", "action"])


def _ids(values) -> str:
    values = list(values)
    return ", ".join(str(v) for v in values) if values else "none"


def clean_patients(
    demographics: pd.DataFrame, diagnosis: pd.DataFrame, log: CleaningLog
) -> pd.DataFrame:
    """Merge demographics with diagnosis and validate the patient table."""
    stage = "Patients"

    duplicated = demographics["pid"][demographics["pid"].duplicated()]
    log.add(stage, "Duplicate participant IDs", f"{len(duplicated)} duplicates ({_ids(duplicated)})")

    patients = demographics.merge(diagnosis, on="pid", how="outer", indicator=True)
    unmatched = patients.loc[patients["_merge"] != "both", "pid"]
    log.add(
        stage,
        "Every participant has exactly one diagnosis",
        f"{len(patients) - len(unmatched)} matched, {len(unmatched)} unmatched ({_ids(unmatched)})",
    )
    patients = patients.drop(columns="_merge")

    patients["sex"] = patients["sex"].str.strip().str.upper()
    invalid_sex = patients.loc[patients["sex"].notna() & ~patients["sex"].isin(["M", "F"]), "pid"]
    log.add(stage, "Sex coded as M/F", f"{len(invalid_sex)} invalid codes ({_ids(invalid_sex)})")

    for column, (low, high) in PLAUSIBLE_RANGES.items():
        values = patients[column]
        outside = patients.loc[values.notna() & ~values.between(low, high), "pid"]
        log.add(
            stage,
            f"{column} within plausible range [{low}, {high}]",
            f"{len(outside)} outside ({_ids(outside)})",
            "Flagged for review" if len(outside) else "None needed",
        )

    patients = _assign_age_group(patients, log)

    # Body-size variables must follow the study design: BMI for adults,
    # weight/height for children. A value in the wrong group would be an entry error.
    for column, group in config.BODY_SIZE_APPLIES_TO.items():
        wrong_group = patients["is_child"] if group == "adults" else ~patients["is_child"]
        misplaced = patients.loc[wrong_group & patients[column].notna(), "pid"]
        log.add(
            stage,
            f"{column} only recorded for {group}",
            f"{len(misplaced)} values in the other group ({_ids(misplaced)})",
            "Set to missing (not applicable)" if len(misplaced) else "None needed",
        )
        patients.loc[wrong_group, column] = np.nan

    return patients.sort_values("pid").reset_index(drop=True)


def _assign_age_group(patients: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Children are younger than 19 (Rocha et al. 2019). Infer the group if age is missing."""
    is_child = patients["age"] < config.CHILD_AGE_LIMIT
    unknown = patients["age"].isna()

    for idx in patients.index[unknown]:
        row = patients.loc[idx]
        if row[["child_weight", "child_height"]].notna().any():
            is_child[idx], reason = True, "has child weight/height"
        elif pd.notna(row["adult_bmi"]):
            is_child[idx], reason = False, "has adult BMI"
        else:
            known = patients.loc[~unknown & (patients["diagnosis"] == row["diagnosis"])]
            children = int((known["age"] < config.CHILD_AGE_LIMIT).sum())
            is_child[idx] = children > len(known) / 2
            reason = (
                f"no body measurements; {children} of {len(known)} other "
                f"{row['diagnosis']} patients are children"
            )
        group = "child" if is_child[idx] else "adult"
        log.add(
            "Patients",
            "Age group for participant with unknown age",
            f"Participant {row['pid']}: {reason}",
            f"Assigned to {group}s",
        )

    patients["is_child"] = is_child.astype(bool)
    patients["age_group"] = np.where(patients["is_child"], "Child", "Adult")
    return patients


def reconcile_recording_names(
    recordings: pd.DataFrame, cycles: pd.DataFrame, log: CleaningLog
) -> pd.DataFrame:
    """Match annotation files whose name differs from the official list only in the device.

    The database's first release had the wrong device in 92 file names; the official
    list carries the corrected name, so the annotation is relabelled to match it.
    """
    listed = set(recordings["recording"])
    by_prefix = {name.rsplit("_", 1)[0]: name for name in listed}
    renames = {
        name: by_prefix[name.rsplit("_", 1)[0]]
        for name in set(cycles["recording"]) - listed
        if name.rsplit("_", 1)[0] in by_prefix
    }
    log.add(
        "Recordings",
        "Annotation file names match the official recording list",
        f"{len(renames)} differ only in the device ({_ids(f'{a} → {b}' for a, b in renames.items())})",
        "Renamed to the corrected official name" if renames else "None needed",
    )
    return cycles.assign(recording=cycles["recording"].replace(renames))


def clean_recordings(
    recordings: pd.DataFrame,
    patients: pd.DataFrame,
    annotation_names: set[str],
    log: CleaningLog,
) -> pd.DataFrame:
    stage = "Recordings"

    for column, allowed in [
        ("location", config.CHEST_LOCATIONS),
        ("mode", config.ACQUISITION_MODES),
        ("device", config.DEVICES),
    ]:
        invalid = recordings.loc[~recordings[column].isin(list(allowed)), "recording"]
        log.add(stage, f"{column} code is documented", f"{len(invalid)} unknown codes ({_ids(invalid)})")

    unknown_pid = recordings.loc[~recordings["pid"].isin(patients["pid"]), "recording"]
    log.add(stage, "Recording belongs to a known participant", f"{len(unknown_pid)} orphans ({_ids(unknown_pid)})")

    listed = set(recordings["recording"])
    no_annotation = sorted(listed - annotation_names)
    unlisted = sorted(annotation_names - listed)
    log.add(
        stage,
        "Every recording has an annotation file",
        f"{len(listed)} listed, {len(no_annotation)} without annotations, "
        f"{len(unlisted)} annotation files not in the list",
        "Recordings without annotations dropped" if no_annotation else "None needed",
    )
    recordings = recordings[recordings["recording"].isin(annotation_names)]

    splits_per_patient = recordings.groupby("pid")["split"].nunique()
    mixed = splits_per_patient[splits_per_patient > 1].index
    log.add(
        stage,
        "Each participant is in a single train/test split",
        f"{len(mixed)} participants in both splits ({_ids(mixed)})",
    )
    return recordings.reset_index(drop=True)


def clean_cycles(cycles: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    stage = "Respiratory cycles"
    cycles = cycles.copy()
    log.add(stage, "Cycles loaded", f"{len(cycles)} cycles in {cycles['recording'].nunique()} recordings")

    for column in ["crackles", "wheezes"]:
        invalid = ~cycles[column].isin([0, 1])
        log.add(stage, f"{column} coded 0/1", f"{int(invalid.sum())} invalid values")

    cycles["duration"] = cycles["end"] - cycles["start"]
    impossible = cycles["duration"] <= 0
    log.add(
        stage,
        "Cycle ends after it starts",
        f"{int(impossible.sum())} impossible cycles",
        "Removed" if impossible.any() else "None needed",
    )
    cycles = cycles[~impossible]

    cycles["truncated"] = cycles["duration"] < MIN_BREATH_SECONDS
    position = cycles.groupby("recording")["start"].rank(method="first")
    at_edge = (position == 1) | (position == cycles.groupby("recording")["start"].transform("size"))
    log.add(
        stage,
        f"Cycle long enough to be a full breath (≥ {MIN_BREATH_SECONDS} s)",
        f"{int(cycles['truncated'].sum())} shorter cycles, "
        f"{int((cycles['truncated'] & at_edge).sum())} of them the first or last cycle of their recording",
        "Kept for crackle/wheeze rates; excluded from cycle-duration features",
    )

    cycles = cycles.sort_values(["recording", "start"])
    previous_end = cycles.groupby("recording")["end"].shift()
    cycles["overlaps_previous"] = cycles["start"] < previous_end - 1e-3
    log.add(
        stage,
        "Cycles do not overlap within a recording",
        f"{int(cycles['overlaps_previous'].sum())} overlapping cycles",
        "Kept and flagged (boundary disagreement, not an invalid breath)"
        if cycles["overlaps_previous"].any()
        else "None needed",
    )
    return cycles.reset_index(drop=True)
