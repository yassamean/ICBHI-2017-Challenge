"""Tests on a small synthetic raw dataset that mimics the ICBHI file formats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icbhi import config, load, missing, noise, preprocess
from icbhi.pipeline import prepare

# pid, age, sex, adult_bmi, child_weight, child_height, diagnosis
PATIENTS = [
    (101, 3, "F", None, 15.0, 95.0, "URTI"),
    (102, 0.75, "F", None, 9.8, 73.0, "Bronchiolitis"),
    (103, 8, "M", None, 26.0, 128.0, "Healthy"),
    (104, 12, "M", None, 40.0, 150.0, "Healthy"),
    (105, 5, "F", None, None, None, "Healthy"),     # genuinely missing weight + height
    (106, 1, "M", None, 11.0, None, "URTI"),        # genuinely missing height
    (107, 15, "F", None, 55.0, 165.0, "Healthy"),
    (108, 70, "F", 28.5, None, None, "COPD"),
    (109, 65, "M", 24.0, None, None, "COPD"),
    (110, 80, "M", 31.0, None, None, "COPD"),
    (111, 72, "F", None, None, None, "COPD"),       # genuinely missing BMI
    (112, 58, "M", 26.0, None, None, "Pneumonia"),
    (113, None, None, None, None, None, "COPD"),    # entire row blank
]


def _na(value):
    return "NA" if value is None else value


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(1)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    demo, diag, split = [], [], []
    for pid, age, sex, bmi, weight, height, diagnosis in PATIENTS:
        demo.append("\t".join(str(_na(v)) for v in (pid, age, sex, bmi, weight, height)))
        diag.append(f"{pid}\t{diagnosis}")
        for index, location in enumerate(["Al", "Pr"]):
            name = f"{pid}_{index + 1}b1_{location}_sc_Meditron"
            split.append(f"{name}\ttrain")
            starts = np.cumsum(rng.uniform(2, 4, size=6))
            rows = [
                f"{s:.3f}\t{s + rng.uniform(1.5, 3.5):.3f}\t{rng.integers(0, 2)}\t{rng.integers(0, 2)}"
                for s in starts
            ]
            (annotations / f"{name}.txt").write_text("\n".join(rows))
    (tmp_path / config.DEMOGRAPHICS_FILE).write_text("\n".join(demo))
    (tmp_path / config.DIAGNOSIS_FILE).write_text("\n".join(diag))
    (tmp_path / config.SPLIT_FILE).write_text("\n".join(split))
    return tmp_path


@pytest.fixture
def prepared(raw_dir):
    return prepare(raw_dir)


def test_parse_recording_name():
    parsed = load.parse_recording_name("101_1b1_Al_sc_Meditron.txt")
    assert parsed == {
        "recording": "101_1b1_Al_sc_Meditron", "pid": 101, "recording_index": "1b1",
        "location": "Al", "mode": "sc", "device": "Meditron",
    }


def test_patient_without_age_is_grouped_and_completed(prepared):
    raw = prepared.patients_raw.set_index("pid")
    done = prepared.patients.set_index("pid")
    assert not raw.at[113, "is_child"]  # COPD patients in the data are adults
    assert np.isnan(raw.at[113, "age"])
    assert done.at[113, "age"] == 71  # median of adult COPD peers (65, 70, 72, 80)
    assert done.at[113, "demographics_imputed"]


def test_missingness_separates_structural_from_genuine(prepared):
    report = missing.missingness_report(prepared.patients_raw).set_index("variable")
    assert report.at["adult_bmi", "not applicable (structural)"] == 7
    assert report.at["adult_bmi", "genuinely missing"] == 2  # 111 and 113
    assert report.at["child_height", "genuinely missing"] == 2  # 105 and 106
    assert report.at["child_weight", "not applicable (structural)"] == 6


def test_listwise_deletion_on_raw_table_removes_everyone(prepared):
    summary = missing.deletion_summary(prepared.patients_raw).set_index("strategy")
    assert summary.at["Listwise deletion on the raw table", "patients kept"] == 0
    assert summary.at["Listwise deletion, structural gaps excused", "patients kept"] == 9


@pytest.mark.parametrize("method", list(missing.METHODS))
def test_imputation_fills_only_genuine_gaps(prepared, method):
    df = prepared.patients
    result = missing.impute(df, ["adult_bmi", "child_weight", "child_height"], method)
    data = result.data.set_index("pid")
    assert data.loc[[105, 106], "child_height"].notna().all()
    assert data.loc[[111, 113], "adult_bmi"].notna().all()
    # structural gaps stay empty
    assert data.loc[~data["is_child"], ["child_weight", "child_height"]].isna().all().all()
    assert data.loc[data["is_child"], "adult_bmi"].isna().all()
    # observed values untouched
    observed = df["child_height"].notna()
    pd.testing.assert_series_equal(result.data.loc[observed, "child_height"], df.loc[observed, "child_height"])


def test_mice_keeps_draws(prepared):
    result = missing.impute(prepared.patients, ["child_height"], "mice", n_imputations=4)
    assert result.draws["child_height"].shape == (2, 4)


def test_subgroup_imputation_reports_empty_groups(prepared):
    result = missing.impute(prepared.patients, ["child_height"], "median", by="diagnosis")
    # 106 is the only URTI patient missing height but 101 is an URTI donor; 105 has Healthy donors
    assert result.imputed["child_height"].sum() == 2
    result = missing.impute(prepared.patients, ["adult_bmi"], "regression", by="sex")
    assert any("left missing" in note for note in result.notes)


def test_whole_vs_subgroup_comparison(prepared):
    table = missing.compare_whole_vs_subgroup(prepared.patients, "child_height", "median", by="sex")
    assert set(table["pid"]) == {105, 106}
    assert {"whole set", "within sex", "difference"} <= set(table.columns)


@pytest.mark.parametrize("subgroup", preprocess.SUBGROUPS)
def test_feature_matrix_is_complete_and_scaled(prepared, subgroup):
    imputed = missing.impute(prepared.patients, missing.IMPUTABLE_COLUMNS[:3], "knn").data
    X, meta, dropped = preprocess.build_feature_matrix(imputed, subgroup)
    assert dropped == []
    assert not X.isna().any().any()
    assert np.allclose(X.mean(), 0, atol=1e-9)
    assert len(meta) == len(X)
    assert "diagnosis" not in X.columns and "main_device" not in X.columns


def test_feature_matrix_drops_constant_columns(prepared):
    imputed = missing.impute(prepared.patients, missing.IMPUTABLE_COLUMNS[:3], "median").data
    X, _, _ = preprocess.build_feature_matrix(imputed, "Adults", sex="M")
    assert "sex_male" not in X.columns


def test_growth_outliers_runs(prepared):
    assert set(noise.growth_outliers(prepared.patients).columns) >= {"pid", "z (for age)"}


def test_sound_shares_are_mutually_exclusive(prepared):
    p = prepared.patients
    cycles = prepared.cycles.merge(prepared.recordings[["recording", "pid"]], on="recording")
    normal = ((cycles["crackles"] == 0) & (cycles["wheezes"] == 0)).groupby(cycles["pid"]).mean()
    total = p.set_index("pid")[["crackle_only_rate", "wheeze_only_rate", "both_rate"]].sum(axis=1) + normal
    assert np.allclose(total, 1.0)
    # the totals include the "both" cycles
    assert np.allclose(p["crackle_rate"], p["crackle_only_rate"] + p["both_rate"])
    assert np.allclose(p["wheeze_rate"], p["wheeze_only_rate"] + p["both_rate"])
    assert not {"crackle_rate", "wheeze_rate"} & set(preprocess.SOUND_FEATURES)
