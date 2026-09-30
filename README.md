# ICBHI 2017 Challenge – Lung Diseases

Course project (data challenges): exploratory analysis, dimensionality reduction and
missing-data imputation on the ICBHI 2017 Respiratory Sound Database and its
demographic data, presented as a Streamlit app.

> Rocha BM et al. (2019) "An open access database for the evaluation of respiratory
> sound classification algorithms", *Physiological Measurement* 40 035001.
> Data: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

The analysis is at **patient level** (126 rows). The audio itself is not used; lung-sound
features come from the expert annotations of the 6,898 respiratory cycles.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Data

The challenge server's HTTPS certificate has expired, so your browser will show a
warning when downloading. Download these files from the challenge page into `data/raw/`:

- `ICBHI_Challenge_demographic_information.txt`
- `ICBHI_Challenge_diagnosis.txt`
- `ICBHI_challenge_train_test.txt`

Then download `ICBHI_final_database.zip` (~2 GB, mostly audio) and copy only the 920
annotation files into the project. Pass the unpacked folder (Safari unzips automatically)
or the zip itself:

```bash
.venv/bin/python scripts/extract_annotations.py ~/Downloads/ICBHI_final_database
```

The download can be deleted afterwards. Build the processed tables and reports:

```bash
.venv/bin/python scripts/build_dataset.py
```

## Run the app

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. Pages: Overview · Data preparation · Statistics ·
Dimensionality reduction · Missing data & imputation (Task 1) · Clustering (Task 2) ·
Classification & imbalance (Task 3). The sidebar settings (patient group, sex, imputation
method, scaling, seed) apply to every page. Each page ends with a *Findings* box.

## Project structure

| Path | Purpose |
|---|---|
| `icbhi/config.py` | Paths, URLs and domain constants (codes, child age limit) |
| `icbhi/load.py` | Parse the raw files |
| `icbhi/clean.py` | Cleaning checks, each logged with finding and action |
| `icbhi/features.py` | Integration: cycles → recordings → one row per patient; derived features |
| `icbhi/missing.py` | Missingness analysis, deletion strategies, imputation methods |
| `icbhi/noise.py` | Outlier / noise identification |
| `icbhi/preprocess.py` | Feature roles, subgroup selection, scaling for dimensionality reduction |
| `icbhi/pipeline.py` | Runs load → clean → integrate → transform |
| `icbhi/reduction.py` | PCA, t-SNE, UMAP, PaCMAP; neighbour preservation and distance correlation |
| `icbhi/clustering.py` | k-means, AGNES, DIANA (own implementation), DBSCAN, OPTICS; Silhouette, intra/inter distances |
| `icbhi/classification.py` | Targets, decision tree / k-NN / random forest, under-/oversampling, repeated stratified CV |
| `app/` | Streamlit app: `streamlit_app.py` (entry, sidebar), `common.py` (caching, charts), `pages/` |
| `tests/` | Tests on synthetic data (`.venv/bin/python -m pytest`) |

## Preprocessing decisions

**Cleaning.** Duplicates, code validity, plausible ranges, the child/adult measurement
rule, file ↔ annotation consistency and cycle validity are checked. Only logically
impossible values are removed; unusual values are flagged. One annotation file still
carries the wrong device name from the database's first release (226_1b1_Pl) and is
matched to its corrected official name. 41 cycles shorter than 0.5 s are all the first or
last cycle of their recording (breaths cut off by the recording edge): they count for the
crackle/wheeze rates but not for the cycle-duration features.

**Unusual values kept.** Participant 179 (10 years, 15 kg, 104 cm) is far below typical
weight and height for age and is flagged by the growth check, but kept as recorded: it may
reflect real undernutrition or a growth disorder. Participant 157 (adult BMI 53.5) is kept
for the same reason.

**Integration.** Cycles are aggregated per patient into the mutually exclusive shares of
cycles with only crackles, only wheezes, or both (the rest are normal cycles, so no cycle
counts twice) and the mean/std of cycle duration, then joined with demographics, diagnosis and
recording metadata.

**Missing values.** Two kinds are distinguished:

- *Structural* – adults have no child weight/height and children no BMI, by study design.
  These are "not applicable" and never imputed. Listwise deletion on the raw table would
  therefore delete every patient.
- *Genuine* – 2 adult BMIs, 5 child weights, 7 child heights and the age/sex of one
  participant. All occur in single-channel Meditron recordings (14 % of Meditron patients,
  0 % elsewhere), so MCAR is implausible and **MAR** given study/device is assumed. MNAR
  cannot be tested with this data.

Participant 223 (entire row blank, COPD) is assigned to the adults (all other COPD
patients are adults) and gets the median age / modal sex of adult COPD patients, because
age and sex are the predictors for every other imputation.

Available imputation methods: median, random sample (hot deck), stochastic regression,
k-nearest neighbours and multiple imputation (MICE). Each can run on the whole set or
separately within a subgroup (sex, age band, diagnosis, …). LOCF does not apply: the data
is cross-sectional.

**Features.** Inputs: age, sex, body size (adult BMI, child weight/height, or BMI z-scored
within age group when adults and children are combined), and the lung-sound features.
Diagnosis, recording device, acquisition mode and split are used for colouring only;
device and mode describe the study, not the patient, and are confounded with age group.

**Normalisation.** Standard (z-score), robust or min–max scaling before dimensionality
reduction, which is distance-based.

**Clustering (Task 2).** Quality uses only the lecture's measures: mean intra- and
inter-cluster distance (Euclidean or Manhattan), within-cluster SSE (elbow) and the
Silhouette coefficient; clusters are compared with age group, sex and diagnosis through
cross-tables and the dominant share per cluster.

**Classification (Task 3).** Targets: 3 diagnosis groups (Chronic / Non-chronic /
Healthy) for all patients, and COPD vs other for adults (64 : 13). Evaluation: repeated
stratified 5-fold cross-validation (the official split has no pneumonia, LRTI or asthma
patients in its test set). Imputation, BMI z-scores, scaling and resampling are fitted on the
training folds only. Undersampling: random undersampling; oversampling: SMOTE (SMOTE-NC
when sex is a feature).
