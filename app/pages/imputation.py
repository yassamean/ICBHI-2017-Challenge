import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (COLOUR_OPTIONS, IMPUTATION_LABELS, INK, NO_IMPUTATION, PALETTE, SYMBOLS, align_to, data,
                    embedding, feature_matrix, findings, imputation, legend_row, scatter, settings,
                    show, show_table, style, theme)
from icbhi import missing, reduction

s = settings()
d = data()
raw = d.patients_raw
VARIABLES = {"child_height": "Child height (cm)", "child_weight": "Child weight (kg)", "adult_bmi": "Adult BMI (kg/m²)"}
METHOD_NOTES = {
    "median": "Fills every gap with the same value. Ignores age, shrinks the variance.",
    "random": "Draws a random observed value. Keeps the distribution, but ignores age.",
    "regression": "Predicts from log(age) and sex and adds random residual noise, so the variance is kept.",
    "knn": "Averages the 5 most similar patients (age, sex, other body measurements).",
    "mice": "Chained regressions, repeated 5 times with random draws; the spread of the draws shows the uncertainty.",
}

st.title("Missing data & imputation")

which, methods_tab, subgroup_tab, effect = st.tabs(
    ["Which features?", "Methods compared", "Whole set vs subgroup", "Effect on the embedding"]
)

# --- a) Which features -----------------------------------------------------------------
with which:
    report = missing.missingness_report(raw)
    st.markdown(
        "Two very different kinds of missing values exist. **Structural** gaps are values that cannot exist: adults "
        "have no child weight/height, children no BMI (study design). They must **not** be imputed. Only **genuinely "
        "missing** values, which should exist but were not recorded, are candidates for imputation."
    )
    fig = go.Figure()
    ink = INK[theme()]
    for column, colour in [("observed", PALETTE[theme()][0]), ("not applicable (structural)", ink["axis"]),
                           ("genuinely missing", PALETTE[theme()][1])]:
        fig.add_trace(go.Bar(y=report["variable"], x=report[column], name=column, orientation="h",
                             marker=dict(color=colour, cornerradius=4),
                             text=report[column].where(report[column] > 0), textposition="inside"))
    style(fig, 260, "Patients per variable (n = 126)")
    fig.update_layout(barmode="stack")
    show(fig)
    st.dataframe(report, width="stretch", hide_index=True)

    st.markdown(
        """
**Where imputation makes sense**
- **Child weight (5) and height (7)** and **adult BMI (2)**: continuous, strongly related to age/sex, few gaps.
  Imputing keeps 9 patients (7 of them children) that deletion would lose.
- **Age and sex** are missing only for participant 223, whose entire row is blank. They are filled once from
  adult COPD patients (median age, most common sex) because every other method uses age and sex as predictors.
- **Not** the structural gaps, and **not** the diagnosis or lung-sound features, which are complete.
"""
    )

    st.markdown("#### Missingness mechanism")
    left, right = st.columns(2)
    with left:
        st.markdown("By main recording device")
        st.dataframe(missing.missingness_by(raw, "main_device"), width="stretch", hide_index=True)
    with right:
        st.markdown("By age group")
        st.dataframe(missing.missingness_by(raw, "age_group"), width="stretch", hide_index=True)
    st.markdown(
        """
- **MCAR** (missing completely at random) is implausible: **all** gaps occur in patients recorded with the Meditron
  stethoscope in single-channel mode (one group of studies), none elsewhere.
- **MAR** (missing at random given observed data) is the working assumption: missingness depends on the study, which is
  observed through the device and age group, and imputation conditions on age and sex.
- **MNAR** (e.g. very ill children not weighed) cannot be ruled out, since the data cannot test it. Participant 223 (whole row
  blank) looks like an administrative gap and is treated as MCAR.
"""
    )
    show_table(missing.genuinely_missing_patients(raw), "Show the 9 patients with genuinely missing values")

    st.markdown("#### Deletion instead of imputation?")
    st.dataframe(missing.deletion_summary(raw), width="stretch", hide_index=True)
    st.markdown(
        "Listwise deletion on the raw table removes **every** patient. Even when structural gaps are excused it drops 9 patients, "
        "7 of them children and 2 of the 26 healthy participants. The loss is not random, so it would bias the remaining sample. "
        "Pairwise deletion is fine for the descriptive statistics (each chart states *n*), but dimensionality reduction, "
        "clustering and classification need a complete table."
    )

# --- b) Methods compared -------------------------------------------------------------------
with methods_tab:
    variable = st.segmented_control("Variable", list(VARIABLES), default="child_height", format_func=VARIABLES.get,
                                    required=True, key="method-variable")
    group_mask = raw["is_child"] if variable != "adult_bmi" else ~raw["is_child"]
    genuine = raw.index[group_mask & raw[variable].isna()]
    methods = list(missing.METHODS)
    results = {m: imputation(m, s.seed) for m in methods}

    fig = go.Figure()
    observed = raw[group_mask & raw[variable].notna()]
    fig.add_trace(go.Scatter(x=observed["age"], y=observed[variable], mode="markers", name="Observed",
                             marker=dict(size=7, color=INK[theme()]["axis"]),
                             customdata=observed[["pid"]], hovertemplate="Patient %{customdata[0]}<br>age %{x}<br>%{y}<extra></extra>"))
    for i, method in enumerate(methods):
        values = results[method].data.loc[genuine]
        error = None
        if method == "mice" and variable in results[method].draws:
            draws = results[method].draws[variable].reindex(genuine)
            error = dict(type="data", symmetric=False, thickness=1.5, width=4,
                         array=(draws.max(axis=1) - values[variable]).to_numpy(),
                         arrayminus=(values[variable] - draws.min(axis=1)).to_numpy())
        fig.add_trace(go.Scatter(
            x=values["age"] + (i - 2) * (0.12 if variable != "adult_bmi" else 0.5), y=values[variable],
            mode="markers", name=missing.METHODS[method], error_y=error,
            marker=dict(size=11, color=PALETTE[theme()][i], symbol=SYMBOLS[i], line=dict(width=1, color=INK[theme()]["surface"])),
            customdata=values[["pid", "age"]],
            hovertemplate=f"<b>{missing.METHODS[method]}</b><br>Patient %{{customdata[0]}}, age %{{customdata[1]}}<br>%{{y:.1f}}<extra></extra>",
        ))
    style(fig, 460, f"Imputed {VARIABLES[variable].lower()} for the {len(genuine)} patients with a genuine gap", legend_rows=2)
    fig.update_xaxes(title_text="Age (years)")
    fig.update_yaxes(title_text=VARIABLES[variable])
    show(fig)
    st.caption("Grey = observed patients. Imputed values are slightly offset horizontally per method so they don't overlap; "
               "MICE shows the range of its 5 draws as an error bar.")

    table = raw.loc[genuine, ["pid", "age", "sex", "diagnosis"]].copy()
    for method in methods:
        table[missing.METHODS[method]] = results[method].data.loc[genuine, variable].round(1)
    st.dataframe(table, width="stretch", hide_index=True)

    summary = pd.DataFrame(
        [{"data": "Observed only", "mean": observed[variable].mean(), "std": observed[variable].std()}]
        + [{"data": missing.METHODS[m], "mean": results[m].data.loc[group_mask, variable].mean(),
            "std": results[m].data.loc[group_mask, variable].std()} for m in methods]
    ).round(2)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Distribution after imputation**")
        st.dataframe(summary, width="stretch", hide_index=True)
    with right:
        st.markdown("**How the methods work**")
        st.markdown("\n".join(f"- **{missing.METHODS[m]}:** {note}" for m, note in METHOD_NOTES.items()))
        st.caption("Last observation carried forward (LOCF) does not apply: the data is cross-sectional, one measurement per patient.")

# --- b) Whole set vs subgroup ------------------------------------------------------------------
with subgroup_tab:
    st.markdown(
        "Does an imputed value change when only a **subgroup** is used as donors instead of the whole set? Child variables "
        "can only be imputed from children anyway (adults have no child height), so subgroups are formed *within* "
        "the applicable patients: by sex, age band or diagnosis."
    )
    controls = st.columns(3)
    variable = controls[0].selectbox("Variable", list(VARIABLES), format_func=VARIABLES.get, key="sub-variable")
    method = controls[1].selectbox("Method", list(missing.METHODS), index=0, format_func=missing.METHODS.get, key="sub-method")
    by = controls[2].selectbox("Subgroup", ["age_band", "sex", "diagnosis_group", "diagnosis"],
                               format_func=lambda b: {"age_band": "Age band", "sex": "Sex", "diagnosis_group": "Diagnosis group",
                                                      "diagnosis": "Diagnosis"}[b])
    whole = imputation(method, s.seed, None, (variable,))
    within = imputation(method, s.seed, by, (variable,))
    rows = whole.imputed[variable] | within.imputed[variable] | (raw[variable].isna() & missing.applicable(raw, variable))
    table = raw.loc[rows, ["pid", "age", "sex", "diagnosis", by]].loc[:, lambda t: ~t.columns.duplicated()].copy()
    table["whole set"] = whole.data.loc[rows, variable]
    table[f"within {by}"] = within.data.loc[rows, variable]
    table["difference"] = table[f"within {by}"] - table["whole set"]

    fig = go.Figure()
    labels = [f"Patient {p} (age {a:g})" for p, a in zip(table["pid"], table["age"])]
    for (label, w, g) in zip(labels, table["whole set"], table[f"within {by}"]):
        fig.add_trace(go.Scatter(x=[w, g], y=[label, label], mode="lines", line=dict(color=INK[theme()]["axis"], width=2),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=table["whole set"], y=labels, mode="markers", name="Whole set",
                             marker=dict(size=11, color=PALETTE[theme()][0])))
    fig.add_trace(go.Scatter(x=table[f"within {by}"], y=labels, mode="markers", name=f"Within {by}",
                             marker=dict(size=11, color=PALETTE[theme()][1], symbol="diamond")))
    style(fig, 120 + 34 * len(table), f"{VARIABLES[variable]}: {missing.METHODS[method]}")
    show(fig)
    st.dataframe(table.round(1), width="stretch", hide_index=True)
    for note in within.notes:
        st.warning(f"Subgroup too small: {note}. With a single subgroup as donors, some gaps cannot be filled at all.")

    st.markdown("#### The extreme case: BMI across adults and children")
    st.markdown(
        "The combined BMI (adult BMI, or weight / height² for children) is the one variable that exists for everyone. "
        "Imputing it from the **whole set** mixes adults and children:"
    )
    whole_bmi = imputation("median", s.seed, None, ("bmi_combined",))
    within_bmi = imputation("median", s.seed, "age_group", ("bmi_combined",))
    rows = whole_bmi.imputed["bmi_combined"]
    bmi_table = raw.loc[rows, ["pid", "age", "age_group", "diagnosis"]].assign(
        **{"whole set": whole_bmi.data.loc[rows, "bmi_combined"], "within age group": within_bmi.data.loc[rows, "bmi_combined"]}
    )
    st.dataframe(bmi_table.round(1), width="stretch", hide_index=True)
    st.caption("Children receive an adult-like BMI from the whole-set median; within their own age group they receive a child value. "
               "This is why BMI is only compared within age groups (z-score) in this project.")

# --- c) Effect on the embedding ------------------------------------------------------------------
with effect:
    st.markdown(
        "The embedding from the dimensionality reduction page computed **without imputation** (listwise deletion: patients with gaps removed) "
        "and **with each imputation method**. Imputed patients are ringed. Every embedding is rotated onto the listwise one "
        "(Procrustes alignment on the shared patients), so positions are comparable."
    )
    controls = st.columns(2)
    method = controls[0].segmented_control("Method", reduction.METHODS, default="PaCMAP", required=True, key="effect-method")
    colour = controls[1].selectbox("Colour by", list(COLOUR_OPTIONS), format_func=COLOUR_OPTIONS.get, key="effect-colour")

    reference = embedding(s.subgroup, s.sex, NO_IMPUTATION, s.scaler, s.seed, method)
    _, meta_ref, dropped = feature_matrix(s.subgroup, s.sex, NO_IMPUTATION, s.scaler, s.seed)
    rows = []
    options = [NO_IMPUTATION, *missing.METHODS]
    legend_row(colour, data().patients[colour])
    columns = st.columns(3) + st.columns(3)
    for column, option in zip(columns, options):
        emb = embedding(s.subgroup, s.sex, option, s.scaler, s.seed, method)
        _, meta, _ = feature_matrix(s.subgroup, s.sex, option, s.scaler, s.seed)
        aligned = emb if option == NO_IMPUTATION else align_to(reference, emb)
        imputed_mask = imputation(option, s.seed).imputed.any(axis=1).reindex(meta.index, fill_value=False)
        frame = pd.concat([aligned, meta], axis=1)
        with column:
            show(scatter(frame, "dim 1", "dim 2", colour, hover=["pid", "age", "sex", "diagnosis"],
                         title=f"{IMPUTATION_LABELS[option]} (n={len(frame)})", height=320, x_title="", y_title="",
                         highlight=imputed_mask, legend=False), key=f"effect-{option}")
        shared = reference.index.intersection(emb.index)
        rows.append({
            "imputation": IMPUTATION_LABELS[option],
            "patients": len(emb),
            "imputed patients": int(imputed_mask.sum()),
            "neighbours kept vs listwise": reduction.neighbour_preservation(reference.loc[shared], emb.loc[shared]),
        })
    rerun = embedding(s.subgroup, s.sex, NO_IMPUTATION, s.scaler, s.seed + 1, method)
    rows.insert(1, {
        "imputation": "None, re-run with another seed (randomness only)",
        "patients": len(rerun),
        "imputed patients": 0,
        "neighbours kept vs listwise": reduction.neighbour_preservation(reference, rerun.loc[reference.index]),
    })
    st.markdown("**How much does the embedding of the shared patients change?**")
    st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
    st.caption("*Neighbours kept vs listwise*: for the patients present in both, the share of each patient's 10 nearest "
               "neighbours in the listwise embedding that are still neighbours after imputation (1 = identical neighbourhoods). "
               "The second row is the reference: the same listwise embedding re-run with another random seed shows how much "
               "changes from the method's own randomness alone.")
    if dropped:
        st.caption(f"Removed by listwise deletion in this selection: patients {', '.join(map(str, dropped))}.")

findings(
    "Most empty cells are **structural** (203 of 219): BMI for children and weight/height for adults do not exist. "
    "Only 16 values are genuinely missing: 14 body measurements in 9 patients, plus the age and sex of participant 223.",
    "All genuine gaps occur in patients recorded with the Meditron stethoscope in single-channel mode, so **MAR** "
    "(dependent on the study) is more plausible than MCAR. Listwise deletion would drop 7 of the 49 children.",
    "Median imputation gives every child the same height (99.5 cm, from 1 to 16 years) and random sampling can give "
    "a 1-year-old 170 cm. Regression, k-NN and MICE follow the growth curve; MICE also shows its uncertainty.",
    "Imputing within a **subgroup** changes values a lot for simple methods (median child height: up to +77 cm "
    "within the 13–18 age band) and can fail in tiny subgroups (only one other LRTI child as donor).",
    "Re-running the embedding with imputation adds 9 patients but changes the shared patients' neighbourhoods only "
    "slightly more than re-running PaCMAP with another random seed: the overall picture is stable.",
)
