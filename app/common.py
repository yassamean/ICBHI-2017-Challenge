"""Shared state, cached computations and chart helpers for all app pages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.linalg import orthogonal_procrustes

from icbhi import missing, preprocess, reduction
from icbhi.pipeline import prepare

BODY_COLUMNS = ["adult_bmi", "child_weight", "child_height"]
NO_IMPUTATION = "none"  # listwise deletion of patients with genuinely missing values
IMPUTATION_LABELS = {NO_IMPUTATION: "None (listwise deletion)", **missing.METHODS}

# --- Settings -----------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    subgroup: str
    sex: str | None
    imputation: str
    scaler: str
    seed: int

    @property
    def label(self) -> str:
        sex = {"F": "female", "M": "male"}.get(self.sex, "")
        group = {"All": "all patients", "Adults": "adults", "Children": "children"}[self.subgroup]
        return f"{sex} {group}".strip()


def sidebar_settings() -> None:
    """Global settings; defined once in the entry script so they persist across pages."""
    st.sidebar.header("Settings")
    st.sidebar.segmented_control("Patients", preprocess.SUBGROUPS, default="All", key="subgroup", required=True)
    st.sidebar.segmented_control(
        "Sex", ["Both", "Female", "Male"], default="Both", key="sex", required=True
    )
    st.sidebar.selectbox(
        "Imputation method",
        list(IMPUTATION_LABELS),
        index=list(IMPUTATION_LABELS).index("mice"),
        format_func=IMPUTATION_LABELS.get,
        key="imputation",
        help="Used wherever a complete data set is needed (dimensionality reduction, clustering, "
        "classification). Structural gaps are never imputed.",
    )
    st.sidebar.selectbox("Scaling", list(preprocess.SCALERS), key="scaler")
    st.sidebar.number_input("Random seed", 0, 9999, 0, key="seed", help="Makes every result reproducible.")
    st.sidebar.caption(
        "Settings apply to every page. Body-size features adapt to the patient group: "
        "BMI for adults, weight and height for children, BMI z-scored within group for all."
    )


def settings() -> Settings:
    state = st.session_state
    return Settings(
        subgroup=state.get("subgroup") or "All",
        sex={"Female": "F", "Male": "M"}.get(state.get("sex")),
        imputation=state.get("imputation", "mice"),
        scaler=state.get("scaler", next(iter(preprocess.SCALERS))),
        seed=int(state.get("seed", 0)),
    )


# --- Cached data ----------------------------------------------------------------------


@st.cache_data(show_spinner="Preparing the data…")
def data():
    return prepare()


@st.cache_data(show_spinner="Imputing missing values…")
def imputation(method: str, seed: int = 0, by: str | None = None, columns: tuple[str, ...] = tuple(BODY_COLUMNS)):
    patients = data().patients
    if method == NO_IMPUTATION:
        return missing.ImputationResult(patients.copy(), pd.DataFrame(False, index=patients.index, columns=list(columns)))
    return missing.impute(patients, list(columns), method, by=by, random_state=seed)


def patients(s: Settings | None = None, method: str | None = None) -> pd.DataFrame:
    """Imputed patient table for the current settings, filtered to the chosen group."""
    s = s or settings()
    table = imputation(method or s.imputation, s.seed).data
    return preprocess.select_subgroup(table, s.subgroup, s.sex)


@st.cache_data(show_spinner=False)
def feature_matrix(subgroup: str, sex: str | None, method: str, scaler: str, seed: int):
    return preprocess.build_feature_matrix(imputation(method, seed).data, subgroup, sex, scaler)


def current_features(s: Settings | None = None, method: str | None = None):
    s = s or settings()
    return feature_matrix(s.subgroup, s.sex, method or s.imputation, s.scaler, s.seed)


@st.cache_data(show_spinner="Computing embedding…")
def embedding(
    subgroup: str, sex: str | None, imputation_method: str, scaler: str, seed: int,
    method: str, n_components: int = 2, perplexity: float = 30.0, n_neighbors: int = 15,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    X, _, _ = feature_matrix(subgroup, sex, imputation_method, scaler, seed)
    if columns:
        X = X[[c for c in columns if c in X.columns]]
    return reduction.embed(X, method, n_components, seed, perplexity, n_neighbors)


def align_to(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Rotate/reflect `other` onto `reference` (on shared rows) so two embeddings are comparable."""
    shared = reference.index.intersection(other.index)
    ref = reference.loc[shared].to_numpy() - reference.loc[shared].to_numpy().mean(axis=0)
    oth = other.loc[shared].to_numpy() - other.loc[shared].to_numpy().mean(axis=0)
    rotation, _ = orthogonal_procrustes(oth, ref)
    centred = other.to_numpy() - other.loc[shared].to_numpy().mean(axis=0)
    aligned = centred @ rotation + reference.loc[shared].to_numpy().mean(axis=0)
    return pd.DataFrame(aligned, index=other.index, columns=other.columns)


# --- Colours and chart styling ------------------------------------------------------------
# Reference palette from the dataviz guidelines; slot order is the colour-blind-safety mechanism.

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "surface": "#fcfcfb"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "surface": "#1a1a19"},
}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "triangle-down"]

# Fixed category orders: a category keeps its colour whatever the filter.
CATEGORY_ORDERS = {
    "diagnosis_group": ["Chronic", "Non-chronic", "Healthy"],
    "diagnosis": ["COPD", "Healthy", "URTI", "Bronchiectasis", "Pneumonia", "Bronchiolitis", "LRTI", "Asthma"],
    "age_group": ["Adult", "Child"],
    "sex": ["F", "M"],
    "main_device": ["Meditron", "AKGC417L", "LittC2SE", "Litt3200"],
    "acquisition_mode": ["sc", "mc", "mc + sc"],
    "split": ["train", "test"],
    "cluster": [f"Cluster {i}" for i in range(1, 9)] + ["Noise"],
}
COLOUR_OPTIONS = {
    "diagnosis_group": "Diagnosis group",
    "diagnosis": "Diagnosis",
    "age_group": "Age group",
    "sex": "Sex",
    "main_device": "Recording device",
}


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def colour_map(column: str, values) -> dict:
    order = CATEGORY_ORDERS.get(column) or sorted(pd.unique(pd.Series(values).dropna()), key=str)
    palette = PALETTE[theme()]
    mapping = {value: palette[i % len(palette)] for i, value in enumerate(order)}
    mapping["Noise"] = INK[theme()]["muted"]
    mapping["Imputed"] = INK[theme()]["primary"]
    return mapping


def ordered(column: str, values) -> list:
    present = set(pd.Series(values).dropna())
    order = CATEGORY_ORDERS.get(column) or sorted(present, key=str)
    return [v for v in order if v in present] + sorted(present - set(order), key=str)


def style(fig: go.Figure, height: int = 380, title: str | None = None, legend: bool = True,
          legend_rows: int = 1) -> go.Figure:
    ink = INK[theme()]
    # Title sits at the top of the figure, the legend in its own row(s) between title and plot.
    top = (32 if title else 8) + (24 * legend_rows + 4 if legend else 0)
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=14, color=ink["primary"]), x=0, xanchor="left",
                   y=1, yanchor="top", yref="container", pad=dict(t=8)) if title else None,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=ink["secondary"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=top, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text="",
                    font=dict(color=ink["secondary"])),
        hoverlabel=dict(font_size=12),
        bargap=0.25,
    )
    axis = dict(automargin=True, gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"],
                tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["secondary"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def scatter(
    frame: pd.DataFrame, x: str, y: str, colour: str, hover: list[str] | None = None,
    title: str | None = None, height: int = 380, highlight: pd.Series | None = None,
    z: str | None = None, x_title: str | None = None, y_title: str | None = None, legend: bool = True,
) -> go.Figure:
    """Scatter coloured by a category; more than three categories also get marker shapes.

    `highlight` (boolean per row) draws a ring around those points (e.g. imputed patients).
    """
    fig = go.Figure()
    categories = ordered(colour, frame[colour])
    colours = colour_map(colour, frame[colour])
    use_symbols = len(categories) > 3
    surface = INK[theme()]["surface"]
    hover = hover or []
    full_order = CATEGORY_ORDERS.get(colour) or categories
    for category in categories:
        position = full_order.index(category) if category in full_order else categories.index(category)
        part = frame[frame[colour] == category]
        custom = part[hover].to_numpy() if hover else None
        template = "<br>".join(f"{h}: %{{customdata[{j}]}}" for j, h in enumerate(hover))
        marker = dict(
            size=9 if z is None else 5, color=colours.get(category), line=dict(width=1, color=surface),
            symbol=SYMBOLS[position % len(SYMBOLS)] if use_symbols else "circle",
        )
        common = dict(name=str(category), mode="markers", marker=marker, customdata=custom,
                      hovertemplate=f"<b>{category}</b><br>{template}<extra></extra>")
        if z is None:
            fig.add_trace(go.Scatter(x=part[x], y=part[y], **common))
        else:
            fig.add_trace(go.Scatter3d(x=part[x], y=part[y], z=part[z], **common))
    if highlight is not None and highlight.any() and z is None:
        marked = frame[highlight.reindex(frame.index, fill_value=False)]
        fig.add_trace(go.Scatter(
            x=marked[x], y=marked[y], mode="markers", name="Imputed patient",
            marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(width=2, color=INK[theme()]["primary"])),
            hoverinfo="skip",
        ))
    style(fig, height, title, legend=legend)
    if z is None:
        fig.update_xaxes(title_text=x_title if x_title is not None else x)
        fig.update_yaxes(title_text=y_title if y_title is not None else y)
    else:
        fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z))
    return fig


def legend_row(colour: str, values) -> None:
    """One shared legend above a grid of small charts (their own legends would wrap into the titles)."""
    categories = ordered(colour, values)
    colours = colour_map(colour, values)
    full_order = CATEGORY_ORDERS.get(colour) or categories
    shapes = ["●", "■", "◆", "▲", "✚", "✖", "★", "▼"]
    items = "".join(
        f'<span style="margin-right:18px;white-space:nowrap"><span style="color:{colours[c]}">'
        f'{shapes[(full_order.index(c) if c in full_order else 0) % 8] if len(categories) > 3 else "●"}</span> {c}</span>'
        for c in categories
    )
    st.markdown(f'<div style="font-size:0.85rem">{items}</div>', unsafe_allow_html=True)


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"scale": 3}})


def show_table(frame: pd.DataFrame, label: str = "Show data table") -> None:
    with st.expander(label):
        st.dataframe(frame, width="stretch", hide_index=True)


def findings(*points: str) -> None:
    """Box of key take-aways at the end of a page (reusable for the presentation slides)."""
    st.markdown("#### Findings")
    st.info("\n".join(f"- {p}" for p in points))

