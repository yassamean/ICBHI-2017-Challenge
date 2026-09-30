"""Run the data preparation pipeline and write the processed tables and reports.

    python scripts/build_dataset.py

Outputs (data/processed/):
    patients_raw.csv        one row per patient, before any imputation
    patients.csv            age/sex completed; body-size gaps still missing
    cycles.csv              cleaned respiratory cycles
    cleaning_log.csv        every cleaning check with its finding and action
    missingness.csv         structural vs genuine missing values per variable
    deletion_summary.csv    patients kept by each deletion strategy
    outliers.csv            IQR and growth-for-age outlier flags
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icbhi import config, missing, noise  # noqa: E402
from icbhi.pipeline import prepare  # noqa: E402


def main() -> None:
    data = prepare()
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    outliers = pd.concat(
        [
            # Crackle/wheeze rates are zero for most patients, so IQR fences collapse to ~0
            # and would flag every patient with any adventitious sound; they are not checked.
            noise.iqr_outliers(
                data.patients, ["cycle_duration_mean", "cycle_duration_std", "adult_bmi"], k=3
            ).assign(check="IQR (k=3)"),
            noise.growth_outliers(data.patients).assign(check="Weight/height for age (|z| > 3)"),
        ],
        ignore_index=True,
    )
    tables = {
        "patients_raw": data.patients_raw,
        "patients": data.patients,
        "cycles": data.cycles,
        "cleaning_log": data.cleaning_log,
        "missingness": missing.missingness_report(data.patients_raw),
        "deletion_summary": missing.deletion_summary(data.patients_raw),
        "outliers": outliers,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(data.cleaning_log.to_string(index=False), end="\n\n")
        print(tables["missingness"].to_string(index=False), end="\n\n")
        print(outliers.to_string(index=False), end="\n\n")
    print(f"Wrote {len(tables)} tables to {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
