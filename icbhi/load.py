"""Readers for the raw ICBHI files. They only parse; all checks happen in clean.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

DEMOGRAPHIC_COLUMNS = ["pid", "age", "sex", "adult_bmi", "child_weight", "child_height"]


def parse_recording_name(name: str) -> dict:
    """Split e.g. '101_1b1_Al_sc_Meditron' into its five documented elements."""
    stem = Path(name).stem
    pid, index, location, mode, device = stem.split("_")
    return {
        "recording": stem,
        "pid": int(pid),
        "recording_index": index,
        "location": location,
        "mode": mode,
        "device": device,
    }


def read_demographics(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / config.DEMOGRAPHICS_FILE,
        sep=r"\s+",
        header=None,
        names=DEMOGRAPHIC_COLUMNS,
        na_values=["NA"],
    )


def read_diagnosis(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / config.DIAGNOSIS_FILE, sep=r"\s+", header=None, names=["pid", "diagnosis"]
    )


def read_recordings(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per recording, from the official train/test list plus the parsed file name."""
    split = pd.read_csv(
        raw_dir / config.SPLIT_FILE, sep=r"\s+", header=None, names=["recording", "split"]
    )
    parsed = pd.DataFrame([parse_recording_name(name) for name in split["recording"]])
    return parsed.assign(split=split["split"].to_numpy())


def read_annotations(annotations_dir: Path = config.ANNOTATIONS_DIR) -> pd.DataFrame:
    """One row per annotated respiratory cycle across all annotation files."""
    frames = []
    for path in sorted(annotations_dir.glob("*.txt")):
        cycles = pd.read_csv(
            path, sep=r"\s+", header=None, names=["start", "end", "crackles", "wheezes"]
        )
        frames.append(cycles.assign(recording=path.stem, cycle=range(len(cycles))))
    if not frames:
        raise FileNotFoundError(
            f"No annotation files in {annotations_dir}. "
            "Run scripts/extract_annotations.py first (see README)."
        )
    columns = ["recording", "cycle", "start", "end", "crackles", "wheezes"]
    return pd.concat(frames, ignore_index=True)[columns]
