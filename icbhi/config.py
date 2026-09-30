"""Paths, source URLs and domain constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
EVENTS_DIR = RAW_DIR / "events"
PROCESSED_DIR = DATA_DIR / "processed"

BASE_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database"
DATABASE_ZIP = "ICBHI_final_database.zip"
DEMOGRAPHICS_FILE = "ICBHI_Challenge_demographic_information.txt"
DIAGNOSIS_FILE = "ICBHI_Challenge_diagnosis.txt"
SPLIT_FILE = "ICBHI_challenge_train_test.txt"
FILENAME_DIFFERENCES_FILE = "filename_differences.txt"
METADATA_FILES = [DEMOGRAPHICS_FILE, DIAGNOSIS_FILE, SPLIT_FILE]

# Rocha et al. (2019): child participants are those "younger than 19 years-old";
# for them weight and height are published instead of BMI.
CHILD_AGE_LIMIT = 19

CHEST_LOCATIONS = {
    "Tc": "Trachea",
    "Al": "Anterior left",
    "Ar": "Anterior right",
    "Pl": "Posterior left",
    "Pr": "Posterior right",
    "Ll": "Lateral left",
    "Lr": "Lateral right",
}
ACQUISITION_MODES = {"sc": "Sequential / single channel", "mc": "Simultaneous / multichannel"}
DEVICES = {
    "AKGC417L": "AKG C417L microphone",
    "LittC2SE": "3M Littmann Classic II SE",
    "Litt3200": "3M Littmann 3200",
    "Meditron": "WelchAllyn Meditron Master Elite",
}
DIAGNOSES = {
    "COPD": "Chronic obstructive pulmonary disease",
    "LRTI": "Lower respiratory tract infection",
    "URTI": "Upper respiratory tract infection",
    "Bronchiectasis": "Bronchiectasis",
    "Bronchiolitis": "Bronchiolitis",
    "Pneumonia": "Pneumonia",
    "Asthma": "Asthma",
    "Healthy": "Healthy",
}

# Grouping used in the ICBHI literature; also the 3-class classification target.
DIAGNOSIS_GROUPS = {
    "COPD": "Chronic",
    "Bronchiectasis": "Chronic",
    "Asthma": "Chronic",
    "URTI": "Non-chronic",
    "LRTI": "Non-chronic",
    "Pneumonia": "Non-chronic",
    "Bronchiolitis": "Non-chronic",
    "Healthy": "Healthy",
}

# Body-size variables only exist for one age group by study design. A missing value
# outside that group is "not applicable" (structural), never a candidate for imputation.
BODY_SIZE_APPLIES_TO = {
    "adult_bmi": "adults",
    "child_weight": "children",
    "child_height": "children",
}
