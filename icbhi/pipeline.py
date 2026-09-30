"""End-to-end data preparation: load → clean → integrate → transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import clean, config, features, load, missing


@dataclass
class PreparedData:
    patients_raw: pd.DataFrame  # one row per patient, before any imputation (for missingness analysis)
    patients: pd.DataFrame  # age/sex completed; body-size gaps not yet imputed
    recordings: pd.DataFrame
    cycles: pd.DataFrame
    cleaning_log: pd.DataFrame


def prepare(raw_dir: Path = config.RAW_DIR) -> PreparedData:
    log = clean.CleaningLog()

    patients = clean.clean_patients(load.read_demographics(raw_dir), load.read_diagnosis(raw_dir), log)
    recordings = load.read_recordings(raw_dir)
    cycles = clean.clean_cycles(load.read_annotations(raw_dir / "annotations"), log)
    cycles = clean.reconcile_recording_names(recordings, cycles, log)
    recordings = clean.clean_recordings(recordings, patients, set(cycles["recording"]), log)

    table = features.build_patient_table(patients, recordings, cycles)
    no_sound = table.loc[table["n_cycles"].isna(), "pid"].tolist()
    log.add(
        "Integration",
        "Every patient has annotated cycles",
        f"{len(table)} patients, {len(no_sound)} without cycles ({no_sound or 'none'})",
    )
    completed = missing.fill_unknown_demographics(table, log)
    return PreparedData(table, completed, recordings, cycles, log.to_frame())
