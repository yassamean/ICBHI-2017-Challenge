import pandas as pd
import streamlit as st

from common import (COLOUR_OPTIONS, Settings, current_features, data, embedding, findings, legend_row, scatter,
                    settings, show, show_table)
from icbhi import reduction

s = settings()
HOVER = ["pid", "age", "sex", "diagnosis", "main_device"]

st.title("Dimensionality reduction")

with st.expander("Which methods are appropriate for this data set?", expanded=False):
    st.markdown(
        """
Following the blog post *"Why you should not rely on t-SNE, UMAP or TriMAP"*:

- **t-SNE** preserves only **local** structure: neighbours stay neighbours, but distances between groups and group sizes
  mean nothing, and results change a lot with *perplexity*. With only 126 patients, perplexity must also stay small.
- **UMAP** is faster and keeps somewhat more global structure, but is still mainly local and depends on *n_neighbors*.
- **TriMAP** tends to struggle with local structure; it is not used here.
- **PaCMAP** uses near, mid-near and far pairs to preserve **local and global** structure and works well with default
  settings, so it is the most appropriate first choice for an unknown data set like ours.
- **PCA** is added as a linear reference: it keeps global structure faithfully but cannot unfold non-linear patterns.

Our data set is small (126 × 8), mixes continuous and binary features, and has one dominant pattern (adults vs children).
PaCMAP and PCA are the most trustworthy here; t-SNE and UMAP are shown for comparison.
"""
    )

controls = st.columns([3, 2, 1, 2, 2, 1])
methods = controls[0].multiselect("Methods", reduction.METHODS, default=reduction.METHODS)
colour = controls[1].selectbox("Colour by", list(COLOUR_OPTIONS), format_func=COLOUR_OPTIONS.get)
dimensions = controls[2].segmented_control("Dimensions", [2, 3], default=2, required=True, format_func=lambda d: f"{d}D")
perplexity = controls[3].slider("t-SNE perplexity", 5, 40, 30)
n_neighbors = controls[4].slider("UMAP / PaCMAP neighbours", 5, 40, 15)
include_sex = controls[5].toggle("Include sex", value=True, help="Sex is a 0/1 feature. After scaling, men and women "
                                 "are about two standard deviations apart, which can dominate the embedding.")


def used_columns(X: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in X.columns if include_sex or c != "sex_male")


def plot(settings_: Settings, method: str, colour_by: str, height: int = 380, title: str | None = None,
         legend: bool = True):
    X, meta, _ = current_features(settings_)
    X = X[list(used_columns(X))]
    emb = embedding(settings_.subgroup, settings_.sex, settings_.imputation, settings_.scaler, settings_.seed,
                    method, dimensions, perplexity, n_neighbors, used_columns(X))
    frame = pd.concat([emb, meta], axis=1)
    fig = scatter(frame, "dim 1", "dim 2", colour_by, hover=HOVER, title=title or method, height=height,
                  z="dim 3" if dimensions == 3 else None, x_title="", y_title="", legend=legend)
    return fig, X, emb


compare, subgroups, sensitivity = st.tabs(["Compare methods", "Subgroups", "Parameter sensitivity"])

with compare:
    if not methods:
        st.info("Select at least one method.")
    rows = []
    columns = st.columns(2)
    for i, method in enumerate(methods):
        fig, X, emb = plot(s, method, colour)
        with columns[i % 2]:
            show(fig, key=f"compare-{method}")
            st.caption(reduction.DESCRIPTIONS[method])
        rows.append({
            "method": method,
            "neighbours kept (local)": reduction.neighbour_preservation(X, emb),
            "distance correlation (global)": reduction.distance_correlation(X, emb),
        })
    if rows:
        st.markdown("**How well is the structure preserved?**")
        st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
        st.caption(
            "*Neighbours kept*: average share of each patient's 10 nearest neighbours (in the full feature space) that are "
            "still among its 10 nearest neighbours in the embedding. *Distance correlation*: rank correlation between all "
            "pairwise distances before and after the reduction. 1 = perfectly preserved."
        )

with subgroups:
    st.markdown(
        "The same method run **separately on each subgroup** (independent of the sidebar selection). Inside a subgroup "
        "the body-size features change (BMI for adults, weight and height for children), and the dominant adult/child "
        "split disappears, so other patterns can surface."
    )
    method = st.segmented_control("Method", reduction.METHODS, default="PaCMAP", required=True, key="sub-method")
    groups = [("All", None, "All patients"), ("Adults", None, "Adults"), ("Children", None, "Children"),
              ("All", "F", "Female"), ("All", "M", "Male")]
    legend_row(colour, data().patients[colour])
    columns = st.columns(3) + st.columns(3)
    for column, (subgroup, sex, title) in zip(columns, groups):
        with column:
            fig, _, _ = plot(Settings(subgroup, sex, s.imputation, s.scaler, s.seed), method, colour, 320, title, False)
            show(fig, key=f"sub-{title}")

with sensitivity:
    st.markdown(
        "The blog post warns that t-SNE and UMAP depend strongly on their hyperparameters, while PaCMAP is robust. "
        "Each row shows one method with four different neighbourhood sizes (2D)."
    )
    X, meta, _ = current_features(s)
    legend_row(colour, meta[colour])
    for method, parameter in [("t-SNE", "perplexity"), ("UMAP", "n_neighbors"), ("PaCMAP", "n_neighbors")]:
        columns = st.columns(4)
        for column, value in zip(columns, [5, 10, 20, 40]):
            emb = embedding(s.subgroup, s.sex, s.imputation, s.scaler, s.seed, method, 2,
                            value if parameter == "perplexity" else perplexity,
                            value if parameter == "n_neighbors" else n_neighbors, used_columns(X))
            frame = pd.concat([emb, meta], axis=1)
            with column:
                show(scatter(frame, "dim 1", "dim 2", colour, hover=HOVER, title=f"{method}, {parameter} = {value}",
                             height=260, x_title="", y_title="", legend=False), key=f"sens-{method}-{value}")

_, meta, dropped = current_features(s)
if dropped:
    st.caption(f"Patients left out because of missing values (no imputation selected): {', '.join(map(str, dropped))}")
show_table(meta, "Show the patients in this embedding")

findings(
    "The dominant structure is **adults vs children**: age separates them, and with them the diagnosis groups "
    "(Chronic = adults, Healthy = children). This is the age/diagnosis confound, not a lung-sound pattern.",
    "Within each age group the islands are split by **sex**: the scaled 0/1 sex feature creates a large distance. "
    "Switch *Include sex* off and the islands merge, showing how one binary feature can dominate distance-based methods.",
    "Healthy and ill children are **not separated** in any method; adults with COPD spread over the whole adult region. "
    "The annotation-based lung-sound features alone do not form clear diagnosis groups.",
    "PCA keeps global distances best (highest distance correlation) but overlaps the groups; t-SNE keeps the most "
    "neighbours but creates islands whose positions carry no meaning; PaCMAP and UMAP show both levels.",
    "All three non-linear methods react to their neighbourhood size, most strongly at very small values (5), where the "
    "data breaks into fragments. The adult/child split is visible in every setting, so it is a robust pattern, not an artefact.",
)
