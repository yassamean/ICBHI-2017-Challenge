import streamlit as st

from common import data

d = data()
p = d.patients

st.title("Lung diseases · ICBHI 2017 Respiratory Sound Database")
st.markdown(
    "Exploration, dimensionality reduction, imputation, clustering and classification of the "
    "[ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) database "
    "combined with its demographic data."
)

cols = st.columns(4)
cols[0].metric("Patients", len(p))
cols[1].metric("Recordings", len(d.recordings))
cols[2].metric("Annotated breathing cycles", f"{len(d.cycles):,}")
cols[3].metric("Diagnoses", p["diagnosis"].nunique())

st.markdown("### The data")
st.markdown(
    f"""
- **Recordings:** 920 lung-sound recordings (10–90 s) from {len(p)} participants, collected by two teams
  in Portugal (Aveiro) and Greece (Thessaloniki), with four different stethoscopes/microphones and
  from seven chest locations.
- **Annotations:** respiratory experts marked every breathing cycle as containing **crackles**, **wheezes**,
  both, or neither. The audio itself is not used here; the lung-sound features come from these annotations.
- **Demographics:** age, sex, and body size: **BMI for adults**, **weight and height for children**
  (younger than 19), because BMI is not comparable between the two.
- **Diagnosis** per participant: COPD, URTI, LRTI, pneumonia, bronchiolitis, bronchiectasis, asthma, or healthy.
"""
)

st.markdown("### Unit of analysis: the patient")
st.markdown(
    "Every row, and every dot in the plots, is **one patient**. Breathing cycles and recordings are "
    "aggregated per patient, so patients with many recordings do not dominate and demographic "
    "statistics count people, not files."
)

st.markdown("### How to use the app")
st.markdown(
    """
| Page | Content |
|---|---|
| Statistics | Distributions by age, sex, diagnosis, body size and lung sounds |
| Dimensionality reduction | PCA, t-SNE, UMAP, PaCMAP side by side; subgroups |
| Missing data & imputation | Missingness, deletion vs imputation, whole set vs subgroup, re-run of the embedding |
| Clustering | k-means, AGNES, DIANA, DBSCAN, OPTICS; quality; feature subsets; imputation |
| Classification & imbalance | Decision tree, k-NN, random forest; under- and oversampling |

The **sidebar settings** (patient group, sex, imputation method, scaling, seed) apply to every page.
"""
)

st.markdown("### Source")
st.caption(
    'Rocha BM et al. (2019) "An open access database for the evaluation of respiratory sound classification '
    'algorithms", *Physiological Measurement* 40 035001.'
)
