import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (INK, NO_IMPUTATION, PALETTE, SEQUENTIAL, data, findings, settings, show,
                    style, theme)
from icbhi import classification, missing
from icbhi.classification import FEATURE_SETS, MODELS, SAMPLER_NOTES, SAMPLERS, TARGETS

s = settings()
BASELINE = next(iter(MODELS))

st.title("Classification & imbalance")

controls = st.columns([2, 2, 3, 1])
target = controls[0].segmented_control("Target", list(TARGETS), default=list(TARGETS)[0], required=True)
feature_set = controls[1].selectbox("Features", list(FEATURE_SETS), help="\n".join(f"- **{k}**: {v}" for k, v in FEATURE_SETS.items()))
models = controls[2].multiselect("Classifiers", list(MODELS), default=list(MODELS))
repeats = controls[3].number_input("Repeats", 2, 20, 10, help="Each repeat is a new random 5-fold split.")

imputation_method = s.imputation
patients = data().patients
if imputation_method == NO_IMPUTATION:
    genuine = missing.genuine_missing_mask(patients, ["adult_bmi", "child_weight", "child_height"]).any(axis=1)
    patients, imputation_method = patients[~genuine], "median"  # nothing left to impute


@st.cache_data(show_spinner="Cross-validating all classifiers and sampling methods (about 20–40 s on first run)…")
def evaluate_grid(target_: str, features_: str, imputation_: str, repeats_: int, seed_: int, listwise: bool):
    """Every classifier × sampling method on the same folds; each fold is imputed once."""
    frame = patients if listwise else data().patients
    results = classification.evaluate_many(frame, target_, features_, list(MODELS), SAMPLERS, imputation_,
                                           n_repeats=repeats_, seed=seed_)
    return {key: (r.folds, r.confusion, r.classes) for key, r in results.items()}


grid = evaluate_grid(target, feature_set, imputation_method, int(repeats), s.seed, s.imputation == NO_IMPUTATION)


def run(model_: str, sampler_: str = "None"):
    return grid[(model_, sampler_)]


dataset, labels = classification.make_dataset(patients, target)
counts = pd.Series(labels).value_counts()
minority = counts.idxmin()
METRICS = ["F1 (macro)", f"F1 ({minority})", "AUC", "accuracy"]


def summary_row(folds: pd.DataFrame) -> dict:
    return {m: f"{folds[m].mean():.2f} ± {folds[m].std():.2f}" for m in METRICS}


def metric_bars(frame: pd.DataFrame, group: str, series: str | None, metric: str, title: str) -> go.Figure:
    """Mean ± standard deviation over all folds; one bar group per `group`, one colour per `series`."""
    fig = go.Figure()
    series_values = [None] if series is None else list(dict.fromkeys(frame[series]))
    for i, value in enumerate(series_values):
        part = frame if value is None else frame[frame[series] == value]
        stats = part.groupby(group, sort=False)[metric].agg(["mean", "std"])
        fig.add_trace(go.Bar(
            x=stats.index, y=stats["mean"], name=str(value) if value else metric,
            error_y=dict(type="data", array=stats["std"], thickness=1.2, width=4, color=INK[theme()]["secondary"]),
            marker=dict(color=PALETTE[theme()][i], cornerradius=4),
            hovertemplate="%{x}<br>mean %{y:.2f}<extra></extra>",
        ))
    style(fig, 350, title, legend=series is not None, legend_rows=2 if series is not None else 1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05], title_text=metric)
    return fig


def confusion_figure(confusion: pd.DataFrame, title: str) -> go.Figure:
    share = confusion.div(confusion.sum(axis=1), axis=0)
    text = [[f"{confusion.iloc[r, c]:.1f}<br>({share.iloc[r, c]:.0%})" for c in range(len(confusion))] for r in range(len(confusion))]
    fig = go.Figure(go.Heatmap(
        z=share.to_numpy(), x=confusion.columns, y=confusion.index, text=text, texttemplate="%{text}",
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)], zmin=0, zmax=1, showscale=False,
        hovertemplate="true %{y} → predicted %{x}<br>%{text}<extra></extra>",
    ))
    style(fig, 300, title, legend=False)
    fig.update_xaxes(title_text="Predicted", side="bottom")
    fig.update_yaxes(title_text="True", autorange="reversed")
    return fig


classify, imbalance, setup = st.tabs(["Classification", "Under- & oversampling", "Evaluation setup"])

# --- Classification ------------------------------------------------------------------------------------------------------------
with classify:
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker=dict(color=PALETTE[theme()][0], cornerradius=4),
                               text=counts.values, textposition="outside"))
        style(fig, 300, f"Classes ({len(labels)} patients)", legend=False)
        fig.update_yaxes(title_text="Patients", range=[0, counts.max() * 1.2])
        show(fig)
        st.caption(f"Imbalance ratio {counts.max() / counts.min():.1f} : 1. Always predicting '{counts.idxmax()}' "
                   f"already gives {counts.max() / counts.sum():.0%} accuracy, which is why accuracy alone is misleading.")
    results = {m: run(m) for m in models}
    with right:
        if results:
            table = pd.DataFrame([{"classifier": m, **summary_row(r[0])} for m, r in results.items()])
            st.markdown(f"**Results** (mean ± std over {int(repeats)} × 5 folds)")
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption(f"**F1 (macro)**: F-measure averaged over the classes, each class counting equally. **F1 ({minority})**: "
                       f"F-measure of the smallest class. **AUC**: area under the ROC curve (one-vs-rest average for 3 classes); "
                       "0.5 = guessing, 1 = perfect.")
    if results:
        folds = pd.concat([r[0].assign(classifier=m) for m, r in results.items()])
        columns = st.columns(2)
        with columns[0]:
            show(metric_bars(folds, "classifier", None, "F1 (macro)", "F-measure (macro)"), key="f1-bars")
        with columns[1]:
            show(metric_bars(folds, "classifier", None, "AUC", "AUC"), key="auc-bars")

        chosen = st.segmented_control("Confusion matrix of", list(results), default=list(results)[-1], required=True)
        columns = st.columns(2)
        with columns[0]:
            show(confusion_figure(results[chosen][1], f"{chosen}: patients per repeat"), key="confusion")
        with columns[1]:
            st.markdown("**Reading the matrix**")
            st.markdown("Rows are the true class, columns the prediction. Numbers are patients per repeat (averaged over "
                        f"{int(repeats)} repeats of the 5 folds); percentages are shares of the true class (recall).")

    st.markdown("#### What the models learned (fitted on all patients, for illustration only)")
    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Decision tree as IF–THEN rules** (thresholds in original units)")
        st.code(classification.tree_rules(patients, target, feature_set, imputation_method, s.seed), language=None)
    with columns[1]:
        importance = classification.feature_importance(patients, target, feature_set, imputation_method, s.seed)
        fig = go.Figure(go.Bar(x=importance.values[::-1], y=importance.index[::-1], orientation="h",
                               marker=dict(color=PALETTE[theme()][0], cornerradius=4)))
        style(fig, 320, "Random forest: feature importance", legend=False)
        fig.update_xaxes(title_text="Mean decrease in impurity")
        show(fig)

# --- Under- & oversampling ------------------------------------------------------------------------------------------------------------
with imbalance:
    st.markdown(
        "Resampling is applied **only to the training folds**; every test fold keeps the real class mix. All variants "
        "use exactly the same folds, so the differences are paired."
    )
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in SAMPLER_NOTES.items()))
    compare_models = st.multiselect("Classifiers to compare", [m for m in MODELS if m != BASELINE],
                                    default=[m for m in MODELS if m != BASELINE], key="imb-models")
    rows, all_folds = [], []
    for model in compare_models:
        for sampler in SAMPLERS:
            folds_, _, _ = run(model, sampler)
            rows.append({"classifier": model, "sampling": sampler, **summary_row(folds_)})
            all_folds.append(folds_.assign(classifier=model, sampling=sampler))
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        folds = pd.concat(all_folds)
        columns = st.columns(2)
        with columns[0]:
            show(metric_bars(folds, "classifier", "sampling", f"F1 ({minority})", f"F-measure of the smallest class ({minority})"),
                 key="imb-minority")
        with columns[1]:
            show(metric_bars(folds, "classifier", "sampling", "AUC", "AUC"), key="imb-auc")

        st.markdown("**Paired change per fold compared with no resampling**")
        base = folds[folds["sampling"] == "None"].set_index(["classifier", "repeat", "fold"])
        diffs = []
        for sampler in SAMPLERS[1:]:
            other = folds[folds["sampling"] == sampler].set_index(["classifier", "repeat", "fold"])
            for metric in ["F1 (macro)", f"F1 ({minority})", "AUC"]:
                change = (other[metric] - base[metric]).groupby(level=0)
                diffs.append(pd.DataFrame({
                    "sampling": sampler, "metric": metric,
                    "mean change": change.mean(), "folds improved": change.apply(lambda x: f"{(x > 0).mean():.0%}"),
                    "folds worse": change.apply(lambda x: f"{(x < 0).mean():.0%}"),
                }))
        st.dataframe(pd.concat(diffs).reset_index().round(3), width="stretch", hide_index=True)
        st.caption("A change is only convincing if it is consistent across folds (most folds improve), not just on average.")

        model = st.segmented_control("Confusion matrices of", compare_models, default=compare_models[-1], required=True,
                                     key="imb-confusion")
        columns = st.columns(3)
        for column, sampler in zip(columns, SAMPLERS):
            with column:
                show(confusion_figure(run(model, sampler)[1], sampler), key=f"cm-{sampler}")

# --- Setup ------------------------------------------------------------------------------------------------------------
with setup:
    st.markdown(
        """
**Targets.** The 8 diagnoses cannot be used directly: asthma has 1 patient and LRTI 2, too few to train, test and oversample.
- **3 groups · all patients**: Chronic (COPD, bronchiectasis, asthma), Non-chronic (URTI, LRTI, pneumonia, bronchiolitis), Healthy.
  Beware: all Chronic patients are adults and all Healthy ones children, so **age alone** separates much of it.
  Compare *Features: Lung sounds only* to see what remains without age.
- **COPD vs other · adults**: removes the age shortcut and is strongly imbalanced (64 : 13), the focus of the under- and oversampling tab.

**Classifiers.** Decision tree (interpretable, depth ≤ 4), k-nearest neighbours (k = 5, distance-based like the clustering) and
random forest (ensemble of 200 trees), plus a baseline that always predicts the majority class. Neural networks were left out:
with 126 patients they would memorise the data.

**Cross-validation instead of the official train/test split.** The patients are split into 5 parts with the same class mix
(stratified); each part is the test set once while the model trains on the other 4. This is repeated with new random splits,
so every patient is tested several times and the spread across folds shows how reliable a difference is.
"""
    )
    split = data().patients.pivot_table(index="diagnosis", columns="split", values="pid", aggfunc="count", fill_value=0)
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(split, width="stretch")
    with right:
        st.markdown(
            f"The official split (made for the audio challenge) has **{int(split.get('test', pd.Series()).sum())} test patients** "
            "and **no pneumonia, LRTI or asthma** patients among them, so those classes could not be evaluated. It also puts "
            "patients 156 and 218 on both sides (per recording, not per patient)."
        )
    st.markdown(
        """
**No information leaks from test to training data.** Inside every fold, in this order and fitted on the training part only:
imputation of body size (training patients are the only donors) → BMI z-score (training means) → scaling → resampling → classifier.

*Limitation:* the age and sex of participant 223 (the one fully blank row) were filled once, before cross-validation, from
other adult COPD patients, i.e. using the diagnosis. This affects one patient's age only.
"""
    )

findings(
    "**3 groups, all features:** random forest and decision tree reach a macro F-measure of about 0.71 (AUC 0.88–0.92) against "
    "0.24 for the majority baseline. But *Chronic* is recognised almost perfectly because all chronic patients are adults: "
    "the models mainly use **age**. Healthy and non-chronic children are often confused.",
    "With **lung sounds only**, performance drops (random forest macro F1 ≈ 0.60, AUC ≈ 0.81): the crackle/wheeze annotations "
    "carry real but limited diagnostic information.",
    "**Adults, COPD vs other (64 : 13):** without resampling the random forest finds only about 2 of the 13 other patients per repeat "
    "(F1 of the minority ≈ 0.20) while accuracy (0.86) looks good, barely above always answering COPD (0.83).",
    "**Undersampling** raises the minority F-measure of all three classifiers (random forest 0.20 → 0.54, decision tree "
    "0.32 → 0.50, k-NN 0.06 → 0.31), at the cost of accuracy (random forest 0.86 → 0.74). **SMOTE** gives smaller gains "
    "(0.27–0.38) with less accuracy loss.",
    "For the adults, **lung sounds alone work as well as all features** (random forest with SMOTE: minority F1 0.52, AUC 0.85): "
    "age, sex and BMI add nothing to telling COPD from other diseases.",
    "With 3 groups (imbalance 2.8 : 1) resampling changes almost nothing. The fold-to-fold spread (± 0.1–0.3) is large with "
    "126 patients, so only consistent paired improvements should be trusted.",
)
