# Data preparation page: commented out on request so it is not shown in the app.
# To show it again, uncomment this file and its st.Page entry in app/streamlit_app.py.
#
# import pandas as pd
# import plotly.graph_objects as go
# import streamlit as st
#
# from common import (PALETTE, current_features, data, findings, scatter, settings, settings_caption,
#                     show, show_table, style, theme)
# from icbhi import noise, preprocess
# from icbhi.preprocess import FEATURE_ROLES
#
# d = data()
# p = d.patients
#
# st.title("Data preparation")
# st.markdown(
#     "Cleaning → integration → transformation → missing values → noise identification → normalisation. "
#     "Every step is computed from the raw files each time the app starts."
# )
#
# cleaning, integration, roles, noise_tab, scaling = st.tabs(
#     ["Cleaning", "Integration", "Feature selection", "Noise identification", "Normalisation"]
# )
#
# with cleaning:
#     st.markdown(
#         "Each check records **what it found** and **what was done**. Only logically impossible values are "
#         "removed; unusual values are flagged and kept."
#     )
#     log = d.cleaning_log
#     stage = st.pills("Stage", list(log["stage"].unique()), selection_mode="multi", default=list(log["stage"].unique()))
#     st.dataframe(log[log["stage"].isin(stage or [])], width="stretch", hide_index=True)
#     st.markdown(
#         """
# **Notable findings**
# - One annotation file still carries the **wrong device name** from the database's first release
#   (`226_1b1_Pl_sc_LittC2SE` → `…_Meditron`); it is matched to its corrected official name instead of being dropped.
# - **41 breathing cycles are shorter than 0.5 s**, which is too short for a real breath. All of them are the first or last cycle of
#   their recording, so they are breaths **cut off by the recording edge**. They count for the crackle/wheeze rates
#   but not for the cycle-duration features.
# - Participant 223 has **no demographic data at all**. They are assigned to the adults (all other COPD patients are adults).
# - Participants 156 and 218 appear in **both** the official train and test split.
# """
#     )
#
# with integration:
#     st.markdown("Four sources are joined on the participant ID and aggregated to **one row per patient**.")
#     cols = st.columns(4)
#     cols[0].metric("Breathing cycles", f"{len(d.cycles):,}")
#     cols[1].metric("Recordings", len(d.recordings))
#     cols[2].metric("Patients", len(p))
#     cols[3].metric("Columns per patient", p.shape[1])
#     st.markdown(
#         """
# | Source | Level | Provides |
# |---|---|---|
# | Annotation files (920) | breathing cycle | start, end, crackles yes/no, wheezes yes/no |
# | File names + train/test list | recording | chest location, acquisition mode, device, split |
# | Demographic file | patient | age, sex, adult BMI, child weight, child height |
# | Diagnosis file | patient | diagnosis |
#
# **Derived per patient:** share of breathing cycles with only crackles, only wheezes, or both (mutually exclusive; the rest
# are normal cycles, so no cycle is counted twice), total crackle and wheeze rates (description only), mean and standard deviation of
# cycle duration, number of recordings, main device; child BMI = weight / height², BMI z-scored within age group,
# age band, diagnosis group (Chronic / Non-chronic / Healthy).
# """
#     )
#     show_table(p, "Show the patient table")
#
# with roles:
#     st.markdown(
#         "Not every attribute should shape the analysis. Attributes describing **how** a patient was recorded "
#         "are kept for colouring only: they are confounded with study, age group and diagnosis "
#         f"(all {int((p['is_child'] & (p['main_device'] == 'Meditron')).sum())} children were recorded with the Meditron)."
#     )
#     st.dataframe(FEATURE_ROLES, width="stretch", hide_index=True)
#     device = pd.crosstab(p["main_device"], p["age_group"])
#     st.markdown("**Recording device × age group**: the device alone almost identifies the study population.")
#     st.dataframe(device, width="stretch")
#
# with noise_tab:
#     st.markdown(
#         "Unusual values are **flagged, not changed**. A plain IQR rule is not meaningful for crackle/wheeze rates "
#         "(zero for most patients, so every non-zero value would be flagged) or for child height (depends on age)."
#     )
#     left, right = st.columns(2)
#     with left:
#         st.markdown("**IQR rule (k = 3)** on BMI and breathing-cycle duration")
#         st.dataframe(
#             noise.iqr_outliers(p, ["adult_bmi", "cycle_duration_mean", "cycle_duration_std"], k=3).round(2),
#             width="stretch", hide_index=True,
#         )
#         st.markdown("**Weight/height for age**: standardised residual of a log(age) growth curve, |z| > 3")
#         st.dataframe(noise.growth_outliers(p), width="stretch", hide_index=True)
#         st.caption(
#             "Participant 179 (10 years, 15 kg, 104 cm) is far below typical growth for their age but is **kept**: "
#             "it may reflect real undernutrition or a growth disorder. Participant 157 (BMI 53.5) is kept as well."
#         )
#     with right:
#         children = p[p["is_child"]].assign(flag=lambda t: t["pid"].eq(179).map({True: "Flagged (179)", False: "Other children"}))
#         fig = scatter(children.dropna(subset=["child_height"]), "age", "child_height", "flag",
#                       hover=["pid", "age", "child_weight", "child_height"], title="Child height by age",
#                       x_title="Age (years)", y_title="Height (cm)")
#         show(fig)
#
#     s = settings()
#     X, meta, _ = current_features(s)
#     scores = noise.isolation_forest_scores(X, random_state=s.seed)
#     st.markdown("**Isolation Forest** (multivariate): patients that are easiest to isolate from the rest of the feature space")
#     settings_caption(s)
#     top = meta.assign(**{"anomaly score": scores}).nlargest(8, "anomaly score")
#     st.dataframe(top[["pid", "age", "sex", "diagnosis", "main_device", "anomaly score"]].round(3),
#                  width="stretch", hide_index=True)
#
# with scaling:
#     s = settings()
#     X, meta, _ = current_features(s)
#     raw = preprocess.add_body_size_features(p).loc[X.index, X.columns]
#     st.markdown(
#         "t-SNE, UMAP, PaCMAP, k-means, k-NN and the other distance-based methods compare features on one scale. Unscaled, "
#         "age (0–93 years) would dominate rates between 0 and 1. The scaler is chosen in the sidebar."
#     )
#     settings_caption(s)
#     left, right = st.columns(2)
#     for column, frame, title in [(left, raw, "Before scaling"), (right, X, f"After: {s.scaler}")]:
#         fig = go.Figure()
#         for feature in frame.columns:
#             fig.add_trace(go.Box(y=frame[feature], name=feature, marker_color=PALETTE[theme()][0],
#                                  line_width=1, boxpoints="outliers", marker_size=4))
#         style(fig, 380, title, legend=False)
#         with column:
#             show(fig)
#
# findings(
#     "The raw data is clean: no duplicates, invalid codes or impossible values; one mislabelled file name and "
#     "41 cut-off breaths are handled explicitly.",
#     "Recording device and acquisition mode describe the study, not the patient. They are excluded as features "
#     "but kept to check for batch effects.",
#     "Scaling is essential: unscaled, age dominates every distance.",
# )
