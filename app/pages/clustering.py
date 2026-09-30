import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage as scipy_linkage

from common import (COLOUR_OPTIONS, IMPUTATION_LABELS, NO_IMPUTATION, PALETTE, Settings, embedding,
                    feature_matrix, findings, imputation, legend_row, scatter, settings, show,
                    show_table, style, theme)
from icbhi import clustering, missing
from icbhi.preprocess import SOUND_FEATURES

s = settings()
SPACES = {
    "All features": None,
    "Lung sounds only": "sound",
    "Demographics & body size": "demographics",
    "PCA (2D)": "PCA",
    "UMAP (2D)": "UMAP",
    "PaCMAP (2D)": "PaCMAP",
}
HOVER = ["pid", "age", "sex", "diagnosis"]

st.title("Clustering")

with st.expander("Quality measures used"):
    st.markdown(
        """
- **Distance d(i, j)**: Euclidean or Manhattan, on the scaled features. k-means always uses Euclidean distance
  (it minimises squared Euclidean distances), and so does AGNES with Ward linkage.
- **Mean intra-cluster distance**: average distance between two patients of the *same* cluster. Smaller is better.
- **Mean inter-cluster distance**: average distance between two patients of *different* clusters. Larger is better.
- **Silhouette coefficient**: for each patient, *a* = mean distance to its own cluster and *b* = mean distance to the nearest
  other cluster; *s = (b − a) / max(a, b)*. The average over all patients lies between −1 and 1: near 1 = well separated,
  near 0 = between clusters, negative = probably in the wrong cluster.
- **Within-cluster sum of squares (SSE)**: what k-means minimises; used for the elbow plot.
- **Noise share** (DBSCAN/OPTICS): patients not assigned to any cluster. They are left out of the distance measures.
- **Hidden patterns**: cross-tables of clusters vs age group, sex and diagnosis, and the **dominant share**: the share of a
  cluster's patients that belong to its most common group (1 = the cluster is exactly one group).
"""
    )

controls = st.columns([2, 2, 2, 1])
space = controls[0].selectbox("Feature space", list(SPACES), help="Clusters are computed in this space.")
metric = controls[1].segmented_control("Distance", list(clustering.METRICS), default="Euclidean", required=True)
compare_to = controls[2].selectbox("Compare clusters with", list(COLOUR_OPTIONS), format_func=COLOUR_OPTIONS.get)
include_sex = controls[3].toggle("Include sex", value=True, help="The 0/1 sex feature can create clusters on its own.")

with st.container(border=True):
    st.markdown("**Method parameters**")
    p = st.columns(5)
    k = p[0].slider("k (k-means, AGNES, DIANA)", 2, 8, 3)
    linkage = p[1].selectbox("AGNES linkage", clustering.LINKAGES, index=0,
                             help="Ward merges the pair of clusters that increases the within-cluster variance least.")
    eps = p[2].slider("DBSCAN eps", 0.2, 4.0, 0.8 if space.endswith("(2D)") else 1.6, 0.1)
    min_samples = p[3].slider("min_samples (DBSCAN, OPTICS)", 3, 15, 5)
    p[4].caption("DBSCAN: a patient with ≥ min_samples neighbours within eps is a core point. OPTICS orders patients "
                 "by reachability and extracts clusters without a fixed eps.")
PARAMS = dict(n_clusters=k, metric=metric, linkage=linkage, eps=eps, min_samples=min_samples, random_state=s.seed)


def space_matrix(settings_: Settings, space_name: str, method_imputation: str | None = None):
    method_imputation = method_imputation or settings_.imputation
    X, meta, dropped = feature_matrix(settings_.subgroup, settings_.sex, method_imputation, settings_.scaler, settings_.seed)
    columns = [c for c in X.columns if include_sex or c != "sex_male"]
    kind = SPACES[space_name]
    if kind == "sound":
        columns = [c for c in columns if c in SOUND_FEATURES]
    elif kind == "demographics":
        columns = [c for c in columns if c not in SOUND_FEATURES]
    X = X[columns]
    if kind in ("PCA", "UMAP", "PaCMAP"):
        X = embedding(settings_.subgroup, settings_.sex, method_imputation, settings_.scaler, settings_.seed, kind,
                      columns=tuple(columns))
    return X, meta, dropped


@st.cache_data(show_spinner="Clustering…")
def run(settings_: Settings, space_name: str, method: str, params: tuple, include: bool, method_imputation: str | None = None):
    X, _, _ = space_matrix(settings_, space_name, method_imputation)
    return clustering.cluster(X, method, **dict(params))


def display_embedding(settings_: Settings, space_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """2D coordinates for plotting: the space itself if 2D, otherwise a PaCMAP projection of it."""
    if X.shape[1] == 2:
        return X.set_axis(["dim 1", "dim 2"], axis=1)
    return embedding(settings_.subgroup, settings_.sex, settings_.imputation, settings_.scaler, settings_.seed,
                     "PaCMAP", columns=tuple(X.columns))


X, meta, dropped = space_matrix(s, space)
labels_by_method = {m: run(s, space, m, tuple(PARAMS.items()), include_sex) for m in clustering.METHODS}

full, choose, subsets, with_imputation = st.tabs(
    ["Clusters & quality", "Choosing parameters", "Feature subsets", "With imputation"]
)

# --- Clusters & quality ------------------------------------------------------------------------------------------------
with full:
    rows = []
    for method, labels in labels_by_method.items():
        quality = clustering.quality(X, labels, metric)
        table = clustering.composition(labels, meta[compare_to])
        clustered = table.drop(index="Noise", errors="ignore")
        rows.append({
            "method": method, "approach": clustering.METHODS[method], **quality,
            f"dominant share ({COLOUR_OPTIONS[compare_to].lower()})":
                (clustered["dominant share"] * clustered["size"]).sum() / max(clustered["size"].sum(), 1),
        })
    quality_table = pd.DataFrame(rows)
    st.markdown(f"**Quality of all methods** · {space}, {metric} distance, {X.shape[1]} features, {len(X)} patients")
    st.dataframe(quality_table.round(3), width="stretch", hide_index=True)
    st.caption("Dominant share here = average over clusters, weighted by cluster size (noise excluded).")

    method = st.segmented_control("Show method", list(clustering.METHODS), default="k-means", required=True)
    labels = labels_by_method[method]
    coords = display_embedding(s, space, X)
    frame = pd.concat([coords, meta], axis=1).assign(cluster=clustering.cluster_names(labels),
                                                      silhouette=np.round(clustering.silhouette_per_point(X, labels, metric), 2))
    left, right = st.columns(2)
    with left:
        show(scatter(frame, "dim 1", "dim 2", "cluster", hover=HOVER + ["silhouette"], title=f"{method} clusters",
                     x_title="", y_title=""), key="clusters")
    with right:
        show(scatter(frame, "dim 1", "dim 2", compare_to, hover=HOVER, title=COLOUR_OPTIONS[compare_to],
                     x_title="", y_title=""), key="groups")
    if X.shape[1] > 2:
        st.caption(f"Clusters are computed in the {X.shape[1]}-dimensional space; the plots use a PaCMAP projection only for display.")

    st.markdown(f"**Composition of the {method} clusters**")
    st.dataframe(clustering.composition(labels, meta[compare_to]).style.format({"dominant share": "{:.0%}"}),
                 width="stretch")
    other = [c for c in ["age_group", "sex", "diagnosis_group"] if c != compare_to]
    columns = st.columns(len(other))
    for column, group in zip(columns, other):
        with column:
            st.markdown(f"vs {COLOUR_OPTIONS[group].lower()}")
            st.dataframe(clustering.composition(labels, meta[group]).style.format({"dominant share": "{:.0%}"}),
                         width="stretch")

# --- Choosing parameters ------------------------------------------------------------------------------------------
with choose:
    left, right = st.columns(2)
    with left:
        method = st.segmented_control("Method", ["k-means", "AGNES", "DIANA"], default="k-means", required=True, key="sweep")
        sweep = clustering.k_sweep(X, method, metric=metric, linkage=linkage, random_state=s.seed)
        fig = go.Figure(go.Scatter(x=sweep["k"], y=sweep["silhouette"], mode="lines+markers",
                                   line=dict(color=PALETTE[theme()][0], width=2), marker=dict(size=8)))
        style(fig, 280, f"{method}: Silhouette by k", legend=False)
        fig.update_xaxes(title_text="k", dtick=1)
        show(fig)
        fig = go.Figure(go.Scatter(x=sweep["k"], y=sweep["within-cluster SSE"], mode="lines+markers",
                                   line=dict(color=PALETTE[theme()][0], width=2), marker=dict(size=8)))
        style(fig, 280, f"{method}: within-cluster SSE by k (elbow)", legend=False)
        fig.update_xaxes(title_text="k", dtick=1)
        show(fig)
        show_table(sweep.round(3), "Show all measures by k")
    with right:
        distances = clustering.k_distance(X, min_samples, metric)
        fig = go.Figure(go.Scatter(y=distances, mode="lines", line=dict(color=PALETTE[theme()][0], width=2)))
        fig.add_hline(y=eps, line=dict(color=PALETTE[theme()][1], width=1.5), annotation_text=f"eps = {eps}",
                      annotation_position="top left")
        style(fig, 280, f"DBSCAN: distance to the {min_samples}-th nearest neighbour (sorted)", legend=False)
        fig.update_xaxes(title_text="Patients (sorted)")
        fig.update_yaxes(title_text="Distance")
        show(fig)
        st.caption("A good eps lies at the 'knee' of this curve: below it most patients are core points, above it the "
                   "distances rise sharply (outliers).")
        dendrogram_linkage = linkage if metric == "Euclidean" or linkage != "ward" else "ward"
        fig = ff.create_dendrogram(
            X.to_numpy(), orientation="bottom", colorscale=PALETTE[theme()],
            linkagefun=lambda data: scipy_linkage(data, method=dendrogram_linkage,
                                                  metric="euclidean" if dendrogram_linkage == "ward" else clustering.METRICS[metric]),
        )
        style(fig, 300, f"AGNES dendrogram ({dendrogram_linkage} linkage)", legend=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(title_text="Merge distance")
        show(fig)
        st.caption("Each merge of two clusters is drawn at the distance where it happens; cutting the tree horizontally "
                   "gives k clusters. Long vertical lines = well-separated clusters.")

# --- Feature subsets ------------------------------------------------------------------------------------------
with subsets:
    st.markdown(
        "The same methods and parameters on different feature spaces. Silhouette values from spaces with different "
        "numbers of dimensions are **not directly comparable** (distances grow with dimension); compare mainly which "
        "groups the clusters capture (dominant share)."
    )
    rows = []
    for space_name in SPACES:
        Xs, meta_s, _ = space_matrix(s, space_name)
        for method in clustering.METHODS:
            labels = run(s, space_name, method, tuple(PARAMS.items()), include_sex)
            quality = clustering.quality(Xs, labels, metric)
            row = {"feature space": space_name, "features": Xs.shape[1], "method": method,
                   "clusters": quality["clusters"], "noise share": quality["noise share"], "silhouette": quality["silhouette"]}
            for group in ["age_group", "sex", "diagnosis_group"]:
                table = clustering.composition(labels, meta_s[group]).drop(index="Noise", errors="ignore")
                row[f"dominant share: {COLOUR_OPTIONS[group].lower()}"] = (
                    (table["dominant share"] * table["size"]).sum() / max(table["size"].sum(), 1)
                )
            rows.append(row)
    subset_table = pd.DataFrame(rows)
    st.dataframe(subset_table.round(2), width="stretch", hide_index=True)
    st.caption("Dominant share of a random split would equal the share of the largest group: "
               f"age group {meta['age_group'].value_counts(normalize=True).max():.2f}, "
               f"sex {meta['sex'].value_counts(normalize=True).max():.2f}, "
               f"diagnosis group {meta['diagnosis_group'].value_counts(normalize=True).max():.2f} (current selection).")

# --- With imputation ------------------------------------------------------------------------------------------
with with_imputation:
    st.markdown(
        "The chosen method clustered after each imputation method, and after listwise deletion. Cluster numbers are "
        "arbitrary, so each result is first matched to the reference clustering (largest overlap) before counting how many "
        "patients **changed cluster**."
    )
    controls = st.columns(2)
    method = controls[0].segmented_control("Method", list(clustering.METHODS), default="k-means", required=True, key="imp-method")
    reference_option = controls[1].selectbox("Reference", list(missing.METHODS), index=list(missing.METHODS).index("mice"),
                                             format_func=missing.METHODS.get)
    reference = pd.Series(run(s, space, method, tuple(PARAMS.items()), include_sex, reference_option),
                          index=space_matrix(s, space, reference_option)[0].index)
    rows, panels = [], []
    for option in [NO_IMPUTATION, *missing.METHODS]:
        Xo, meta_o, _ = space_matrix(s, space, option)
        labels = pd.Series(run(s, space, method, tuple(PARAMS.items()), include_sex, option), index=Xo.index)
        shared = reference.index.intersection(labels.index)
        aligned = pd.Series(clustering.align_labels(reference.loc[shared].to_numpy(), labels.loc[shared].to_numpy()), index=shared)
        imputed = imputation(option, s.seed).imputed.any(axis=1).reindex(Xo.index, fill_value=False)
        moved = aligned != reference.loc[shared]
        rows.append({
            "imputation": IMPUTATION_LABELS[option], "patients": len(Xo),
            **{k_: v for k_, v in clustering.quality(Xo, labels.to_numpy(), metric).items() if k_ != "clusters"},
            "changed cluster vs reference": int(moved.sum()),
            "of which imputed patients": int((moved & imputed.reindex(shared, fill_value=False)).sum()),
        })
        panels.append((option, Xo, meta_o, labels, imputed))
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)

    st.markdown("**Clusters per imputation method** (imputed patients ringed; shown on a PaCMAP projection of the reference data)")
    reference_X = space_matrix(s, space, reference_option)[0]
    coords = display_embedding(Settings(s.subgroup, s.sex, reference_option, s.scaler, s.seed), space, reference_X)
    legend_row("cluster", [f"Cluster {i + 1}" for i in range(k)] + ["Noise"])
    columns = st.columns(3) + st.columns(3)
    for column, (option, Xo, meta_o, labels, imputed) in zip(columns, panels):
        shared = coords.index.intersection(labels.index)
        aligned = clustering.align_labels(reference.loc[shared].to_numpy(), labels.loc[shared].to_numpy())
        frame = pd.concat([coords.loc[shared], meta_o.loc[shared]], axis=1).assign(cluster=clustering.cluster_names(aligned))
        with column:
            show(scatter(frame, "dim 1", "dim 2", "cluster", hover=HOVER, title=IMPUTATION_LABELS[option], height=300,
                         highlight=imputed.loc[shared], legend=False, x_title="", y_title=""), key=f"imp-{option}")

findings(
    "With default settings (all patients, all features, k = 3) **k-means and AGNES essentially rediscover the age groups** "
    "(93 % / 87 % of a cluster share one age group vs 61 % by chance), and through age the diagnosis groups.",
    "The Silhouette of k-means, AGNES and DIANA is low (≈ 0.15–0.3 for every k): the patients do not form well-separated "
    "clusters. DBSCAN and OPTICS reach somewhat higher values (0.32 / 0.38) only because they declare 42–60 % of the patients noise.",
    "Density-based clusters split **exactly by sex** (dominant share 1.0): the binary sex feature creates two dense "
    "layers. With *Include sex* off, the clusters follow the age groups instead.",
    "On **lung sounds only**, clusters no longer follow age, sex or diagnosis, except one cluster of 16 patients, all "
    "chronic, with many wheeze-only cycles. On a **PaCMAP** projection Silhouettes rise to about 0.6, because the embedding "
    "compacts the groups, not because the clusters are better.",
    "**Imputation and k-means:** the children's cluster is identical in every variant. With regression, k-NN or MICE no patient "
    "changes cluster; with median, random sample or listwise deletion about 30 patients do, almost all of them adults whose own values "
    "were not imputed. The adults have no natural subdivision, so a small change in the data flips how k-means splits them.",
)
