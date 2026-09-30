import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK, colour_map, data, findings, ordered, scatter, settings, show, show_table, style,
                    theme)
from icbhi import preprocess

s = settings()
# Statistics describe observed data, so they use the un-imputed table (pairwise deletion: each
# chart uses every patient that has the variable, and states n).
p = preprocess.select_subgroup(data().patients_raw, s.subgroup, s.sex)
# Charts by age group only use patients whose age is known (participant 223 has no age).
known_age = p[p["age"].notna()]

st.title("Statistics")

cols = st.columns(4)
cols[0].metric("Patients", len(p))
cols[1].metric("Children (< 19)", int(p["is_child"].sum()))
cols[2].metric("Female", int((p["sex"] == "F").sum()))
cols[3].metric("Median age", f"{p['age'].median():g} years")


def histogram(frame: pd.DataFrame, column: str, colour: str, title: str, x_title: str, nbins: int = 20) -> go.Figure:
    fig = go.Figure()
    colours = colour_map(colour, frame[colour])
    for category in ordered(colour, frame[colour]):
        values = frame.loc[frame[colour] == category, column].dropna()
        fig.add_trace(go.Histogram(x=values, name=f"{category} (n={len(values)})", nbinsx=nbins,
                                   marker=dict(color=colours[category], line=dict(width=1, color=INK[theme()]["surface"]))))
    style(fig, 320, title)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Patients")
    return fig


def bars(counts: pd.DataFrame, title: str, colour_column: str, horizontal: bool = False) -> go.Figure:
    """Stacked bars: rows = categories on the axis, columns = colour groups."""
    fig = go.Figure()
    colours = colour_map(colour_column, counts.columns)
    for group in ordered(colour_column, counts.columns):
        values = counts[group]
        kwargs = dict(y=counts.index, x=values, orientation="h") if horizontal else dict(x=counts.index, y=values)
        fig.add_trace(go.Bar(name=str(group), marker=dict(color=colours[group], cornerradius=4), **kwargs))
    style(fig, 340, title)
    fig.update_layout(barmode="stack")
    return fig


st.markdown("### Age and sex")
left, right = st.columns(2)
with left:
    adults, children = p[~p["is_child"]], p[p["is_child"]]
    tab_children, tab_adults = st.tabs(["Children", "Adults"])
    with tab_children:
        show(histogram(children, "age", "sex", f"Age of children (n={children['age'].notna().sum()})", "Age (years)", 16))
    with tab_adults:
        show(histogram(adults, "age", "sex", f"Age of adults (n={adults['age'].notna().sum()})", "Age (years)", 16))
with right:
    counts = pd.crosstab(known_age["age_group"], known_age["sex"]).reindex(columns=ordered("sex", known_age["sex"]))
    fig = bars(counts, f"Sex by age group (n={int(counts.to_numpy().sum())})", "sex")
    fig.update_yaxes(title_text="Patients")
    show(fig)

st.markdown("### Diagnosis")
BY = {"age_group": ("By age group", known_age), "sex": ("By sex", p[p["sex"].notna()])}
left, right = st.columns(2)
with left:
    for tab, (column, (label, frame)) in zip(st.tabs([label for label, _ in BY.values()]), BY.items()):
        with tab:
            counts = pd.crosstab(frame["diagnosis"], frame[column])
            counts = counts.loc[counts.sum(axis=1).sort_values().index]
            fig = bars(counts, f"Diagnoses {label.lower()} (n={len(frame)})", column, horizontal=True)
            fig.update_xaxes(title_text="Patients")
            show(fig, key=f"diagnoses-{column}")
with right:
    for tab, (column, (label, frame)) in zip(st.tabs([label for label, _ in BY.values()]), BY.items()):
        with tab:
            counts = pd.crosstab(frame["diagnosis_group"], frame[column]).reindex(
                ordered("diagnosis_group", frame["diagnosis_group"]))
            fig = bars(counts, f"Diagnosis groups {label.lower()} (n={len(frame)})", column)
            fig.update_yaxes(title_text="Patients")
            show(fig, key=f"groups-{column}")

st.markdown("### Body size")
left, right = st.columns(2)
with left:
    bmi = p[~p["is_child"]]
    show(histogram(bmi, "adult_bmi", "sex", f"Adult BMI (n={bmi['adult_bmi'].notna().sum()})", "BMI (kg/m²)", 20))
with right:
    kids = p[p["is_child"]].dropna(subset=["child_height"])
    if len(kids):
        show(scatter(kids, "age", "child_height", "sex", hover=["pid", "age", "child_weight", "child_height"],
                     title=f"Child height by age (n={len(kids)})", height=320, x_title="Age (years)", y_title="Height (cm)"))
    else:
        st.info("No children in the current selection.")

st.markdown("### Lung sounds")
st.markdown("Share of each patient's breathing cycles with crackles or wheezes, and the mean length of a breathing cycle. "
            "*All* counts every cycle with that sound; *only* and *crackles + wheezes* split the cycles without overlap.")
feature = st.segmented_control(
    "Feature", ["crackle_rate", "wheeze_rate", "crackle_only_rate", "wheeze_only_rate", "both_rate", "cycle_duration_mean"],
    default="crackle_rate",
    format_func=lambda f: {"crackle_rate": "Crackles (all)", "wheeze_rate": "Wheezes (all)",
                           "crackle_only_rate": "Crackles only", "wheeze_only_rate": "Wheezes only",
                           "both_rate": "Crackles + wheezes", "cycle_duration_mean": "Cycle duration (s)"}[f],
    required=True,
)
fig = go.Figure()
colours = colour_map("diagnosis", p["diagnosis"])
for diagnosis in ordered("diagnosis", p["diagnosis"]):
    values = p.loc[p["diagnosis"] == diagnosis, feature]
    fig.add_trace(go.Box(y=values, name=f"{diagnosis} (n={len(values)})", marker=dict(color=colours[diagnosis], size=5),
                         line_width=1.5, boxpoints="all", jitter=0.4, pointpos=0))
style(fig, 380, legend=False)
show(fig)

st.markdown("### Recordings")
left, right = st.columns(2)
with left:
    counts = pd.crosstab(known_age["main_device"], known_age["age_group"])
    fig = bars(counts, "Main recording device by age group", "age_group")
    fig.update_yaxes(title_text="Patients")
    show(fig)
with right:
    show(histogram(known_age, "n_recordings", "age_group", "Recordings per patient", "Recordings", 20))
    st.caption("Adults from the multichannel studies have many more recordings, which is why the analysis is per patient.")

show_table(p.drop(columns=["sex_male"]), "Show the patient table for this selection")

unknown = p.loc[p["age"].isna() | p["sex"].isna(), "pid"].astype(int).tolist()
findings(
    f"{len(p)} patients in this selection: {int((p['age'] >= 19).sum())} adults and {int((p['age'] < 19).sum())} children, "
    f"{int((p['sex'] == 'M').sum())} male and {int((p['sex'] == 'F').sum())} female"
    + (f"; age and sex are missing for participant {', '.join(map(str, unknown))}." if unknown else "."),
    "Age is bimodal: children are mostly very young (median 3 years), adults mostly elderly (median 69 years).",
    "Diagnosis is strongly imbalanced (COPD 64 vs asthma 1) and confounded with age: healthy = children, COPD = adults.",
    "Lung sounds differ by diagnosis: COPD patients show the most crackles, asthma and bronchiolitis the most wheezes.",
    "All children were recorded with the Meditron stethoscope, so device, study and age group coincide.",
)
