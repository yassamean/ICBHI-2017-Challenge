"""Missing-data analysis, deletion strategies and imputation.

Two kinds of missing values exist in this dataset:

* Structural ("not applicable"): adults have no child weight/height and children
  have no adult BMI, by study design. These values do not exist and are never imputed.
* Genuine: a value that should exist but was not recorded. Only these are imputed,
  and only within the population the variable applies to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression

from . import config
from .clean import CleaningLog

REPORT_COLUMNS = ["age", "sex", "adult_bmi", "child_weight", "child_height"]
IMPUTABLE_COLUMNS = ["adult_bmi", "child_weight", "child_height", "bmi_combined"]
PREDICTORS = ["log_age", "sex_male"]

METHODS = {
    "median": "Median",
    "random": "Random sample (hot deck)",
    "regression": "Stochastic regression",
    "knn": "k-nearest neighbours",
    "mice": "Multiple imputation (MICE)",
}
# Minimum number of observed values in a (sub)group before a method is attempted.
MIN_DONORS = {"median": 1, "random": 1, "regression": len(PREDICTORS) + 2, "knn": 1, "mice": 3}


def applicable(df: pd.DataFrame, column: str) -> pd.Series:
    """Rows for which `column` can exist at all (see config.BODY_SIZE_APPLIES_TO)."""
    group = config.BODY_SIZE_APPLIES_TO.get(column)
    if group == "adults":
        return ~df["is_child"]
    if group == "children":
        return df["is_child"].copy()
    return pd.Series(True, index=df.index)


# --- Analysis -----------------------------------------------------------------------


def missingness_report(df: pd.DataFrame, columns: list[str] = REPORT_COLUMNS) -> pd.DataFrame:
    rows = []
    for column in columns:
        applies = applicable(df, column)
        missing = df[column].isna()
        rows.append(
            {
                "variable": column,
                "applies to": config.BODY_SIZE_APPLIES_TO.get(column, "everyone"),
                "observed": int((~missing).sum()),
                "not applicable (structural)": int((missing & ~applies).sum()),
                "genuinely missing": int((missing & applies).sum()),
                "% missing of applicable": round(100 * (missing & applies).sum() / applies.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def genuine_missing_mask(df: pd.DataFrame, columns: list[str] = REPORT_COLUMNS) -> pd.DataFrame:
    return pd.DataFrame({c: df[c].isna() & applicable(df, c) for c in columns})


def genuinely_missing_patients(df: pd.DataFrame, columns: list[str] = REPORT_COLUMNS) -> pd.DataFrame:
    """Patients with at least one genuinely missing value, with context for judging the mechanism."""
    mask = genuine_missing_mask(df, columns)
    affected = df[mask.any(axis=1)].copy()
    affected["missing"] = mask[mask.any(axis=1)].apply(lambda r: ", ".join(r.index[r]), axis=1)
    return affected[
        ["pid", "age", "sex", "age_group", "diagnosis", "devices", "acquisition_mode", "n_recordings", "missing"]
    ]


def missingness_by(df: pd.DataFrame, by: str, columns: list[str] = REPORT_COLUMNS) -> pd.DataFrame:
    """Share of patients with any genuinely missing value per category of `by`.

    If missingness concentrates in one observed category (e.g. one device/study),
    MCAR is implausible and MAR given that category is the working assumption.
    """
    has_missing = genuine_missing_mask(df, columns).any(axis=1)
    return (
        pd.DataFrame({by: df[by], "has_missing": has_missing})
        .groupby(by)["has_missing"]
        .agg(patients="size", with_missing="sum")
        .assign(**{"% with missing": lambda t: (100 * t["with_missing"] / t["patients"]).round(1)})
        .reset_index()
    )


def deletion_summary(df: pd.DataFrame, columns: list[str] = REPORT_COLUMNS) -> pd.DataFrame:
    """How many patients each deletion strategy keeps."""

    def describe(strategy: str, kept: pd.Series, note: str) -> dict:
        return {
            "strategy": strategy,
            "patients kept": int(kept.sum()),
            "patients lost": int((~kept).sum()),
            "children kept": int((kept & df["is_child"]).sum()),
            "adults kept": int((kept & ~df["is_child"]).sum()),
            "healthy kept": int((kept & (df["diagnosis"] == "Healthy")).sum()),
            "note": note,
        }

    raw_complete = df[columns].notna().all(axis=1)
    applicable_complete = ~genuine_missing_mask(df, columns).any(axis=1)
    rows = [
        describe("No deletion (all patients)", pd.Series(True, index=df.index), "Reference"),
        describe(
            "Listwise deletion on the raw table",
            raw_complete,
            "Every patient lacks BMI or weight/height by design",
        ),
        describe(
            "Listwise deletion, structural gaps excused",
            applicable_complete,
            "Only genuinely missing values cause deletion",
        ),
    ]
    for column in columns:
        kept = df[column].notna() | ~applicable(df, column)
        rows.append(
            describe(f"Pairwise: analyses using {column}", kept, "Sample size differs per variable")
        )
    return pd.DataFrame(rows)


# --- Imputation -----------------------------------------------------------------------


def fill_unknown_demographics(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Fill missing age/sex from patients with the same diagnosis and age group.

    Age and sex are predictors for every other imputation, so they are completed
    first. In this dataset this affects a single participant whose entire row is
    blank (treated as MCAR); keeping them preserves their annotated recordings.
    """
    df = df.copy()
    df["demographics_imputed"] = df["age"].isna() | df["sex"].isna()
    for idx in df.index[df["demographics_imputed"]]:
        peers = df[
            (df["diagnosis"] == df.at[idx, "diagnosis"])
            & (df["is_child"] == df.at[idx, "is_child"])
            & ~df["demographics_imputed"]
        ]
        filled = []
        if pd.isna(df.at[idx, "age"]):
            df.at[idx, "age"] = peers["age"].median()
            filled.append(f"age = {df.at[idx, 'age']:g} (median)")
        if pd.isna(df.at[idx, "sex"]):
            df.at[idx, "sex"] = peers["sex"].mode().iloc[0]
            filled.append(f"sex = {df.at[idx, 'sex']} (mode)")
        log.add(
            "Missing values",
            "Complete age/sex (predictors for all other imputations)",
            f"Participant {df.at[idx, 'pid']} has no age/sex",
            f"Filled from {len(peers)} {df.at[idx, 'age_group'].lower()} {df.at[idx, 'diagnosis']} "
            f"patients: {', '.join(filled)}",
        )
    df["sex_male"] = df["sex"].map({"M": 1.0, "F": 0.0})
    return df


@dataclass
class ImputationResult:
    data: pd.DataFrame
    imputed: pd.DataFrame  # True where a value was filled in
    draws: dict[str, pd.DataFrame] = field(default_factory=dict)  # MICE: m values per patient
    notes: list[str] = field(default_factory=list)


def impute(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    by: str | None = None,
    random_state: int = 0,
    n_neighbors: int = 5,
    n_imputations: int = 5,
    donors: pd.Index | None = None,
) -> ImputationResult:
    """Impute genuinely missing values of `columns` with `method`.

    With `by`, the method is run separately inside each category of `by`
    (e.g. sex or age band), using only that subgroup as donors. Structural gaps are
    never filled because each column is only imputed within its applicable rows.
    Age and sex must be complete (see fill_unknown_demographics); they are the predictors.
    With `donors`, only those rows provide observed values / fit the model (used inside
    cross-validation so that test patients never inform the imputation).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    if df[["age", "sex_male"]].isna().any().any():
        raise ValueError("Age and sex must be complete before imputation")

    work = df.copy()
    work["log_age"] = np.log1p(work["age"])
    rng = np.random.default_rng(random_state)
    result = ImputationResult(
        data=df.copy(), imputed=pd.DataFrame(False, index=df.index, columns=columns)
    )

    donors = df.index if donors is None else pd.Index(donors)
    groups = {"all patients": df.index} if by is None else df.groupby(by).groups
    for group_name, group_index in groups.items():
        label = "All patients" if by is None else f"{by} = {group_name}"
        for column in columns:
            rows = pd.Index(group_index).intersection(df.index[applicable(df, column)])
            observed = rows[work.loc[rows, column].notna()].intersection(donors)
            missing = rows[work.loc[rows, column].isna()]
            if missing.empty:
                continue
            if len(observed) < MIN_DONORS[method]:
                result.notes.append(
                    f"{label}: {column} left missing for {len(missing)} patient(s), "
                    f"only {len(observed)} observed value(s) in this group"
                )
                continue

            if method in ("knn", "mice"):
                values, draws = _impute_multivariate(
                    work.loc[rows], column, method, n_neighbors, n_imputations, random_state,
                    rows.intersection(donors),
                )
                values = values.loc[missing]
                if draws is not None:
                    result.draws.setdefault(column, [])
                    result.draws[column].append(draws.loc[missing])
            else:
                values = _impute_univariate(work, column, observed, missing, method, rng)

            result.data.loc[missing, column] = values.to_numpy()
            result.imputed.loc[missing, column] = True

    result.draws = {c: pd.concat(d) for c, d in result.draws.items()}
    return result


def _impute_univariate(work, column, observed, missing, method, rng) -> pd.Series:
    y = work.loc[observed, column].to_numpy(dtype=float)
    if method == "median":
        values = np.full(len(missing), np.median(y))
    elif method == "random":
        values = rng.choice(y, size=len(missing), replace=True)
    else:  # stochastic regression on log(age) and sex, plus residual noise
        x_obs = work.loc[observed, PREDICTORS].to_numpy()
        model = LinearRegression().fit(x_obs, y)
        residual_sd = np.std(y - model.predict(x_obs), ddof=len(PREDICTORS) + 1)
        values = model.predict(work.loc[missing, PREDICTORS].to_numpy())
        values = values + rng.normal(0, residual_sd, size=len(missing))
        values = np.clip(values, y.min() * 0.5, None)  # no negative weights/heights/BMIs
    return pd.Series(values, index=missing)


def _impute_multivariate(sub, column, method, n_neighbors, n_imputations, random_state, donors):
    """KNN / MICE on [target, other variables of the same population, predictors].

    The imputer is fitted on the donor rows only and then applied to all rows of `sub`.
    """
    siblings = [
        c
        for c, group in config.BODY_SIZE_APPLIES_TO.items()
        if c != column
        and group == config.BODY_SIZE_APPLIES_TO.get(column)
        and sub.loc[donors, c].notna().sum() >= MIN_DONORS[method]
    ]
    features = [column, *siblings, *PREDICTORS]
    matrix = sub[features].to_numpy(dtype=float)
    fit_rows = sub.index.get_indexer(donors)

    # Standardise (with donor statistics) so no variable dominates by its unit.
    mean = np.nanmean(matrix[fit_rows], axis=0)
    std = np.nanstd(matrix[fit_rows], axis=0)
    std[std == 0] = 1.0
    scaled = (matrix - mean) / std

    if method == "knn":
        n_donors = int(sub.loc[donors, column].notna().sum())
        imputer = KNNImputer(n_neighbors=min(n_neighbors, n_donors)).fit(scaled[fit_rows])
        filled = imputer.transform(scaled)[:, 0] * std[0] + mean[0]
        return pd.Series(filled, index=sub.index), None

    draws = []
    for i in range(n_imputations):
        imputer = IterativeImputer(sample_posterior=True, max_iter=15, random_state=random_state + i)
        draw = imputer.fit(scaled[fit_rows]).transform(scaled)[:, 0] * std[0] + mean[0]
        draws.append(np.clip(draw, np.nanmin(matrix[fit_rows, 0]) * 0.5, None))
    draws = pd.DataFrame(
        np.column_stack(draws), index=sub.index, columns=[f"draw {i + 1}" for i in range(n_imputations)]
    )
    return draws.mean(axis=1), draws


def compare_whole_vs_subgroup(
    df: pd.DataFrame, column: str, method: str, by: str, random_state: int = 0
) -> pd.DataFrame:
    """Impute `column` once using all patients and once within each `by` subgroup."""
    whole = impute(df, [column], method, random_state=random_state)
    within = impute(df, [column], method, by=by, random_state=random_state)
    rows = whole.imputed[column] | within.imputed[column]
    table = df.loc[rows, ["pid", "age", "sex", "age_group", "diagnosis", by]].copy()
    table = table.loc[:, ~table.columns.duplicated()]
    table["whole set"] = whole.data.loc[rows, column]
    table[f"within {by}"] = within.data.loc[rows, column]
    table["difference"] = table[f"within {by}"] - table["whole set"]
    return table.reset_index(drop=True)
