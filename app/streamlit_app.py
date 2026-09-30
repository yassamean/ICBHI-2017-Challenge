"""ICBHI 2017 lung-disease analysis app.  Run:  .venv/bin/python -m streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from common import sidebar_settings  # noqa: E402

st.set_page_config(page_title="ICBHI 2017 · Lung Diseases", page_icon="🫁", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True),
    # st.Page("pages/preparation.py", title="Data preparation", icon=":material/cleaning_services:"),
    st.Page("pages/statistics.py", title="Statistics", icon=":material/bar_chart:"),
    st.Page("pages/reduction.py", title="Dimensionality reduction", icon=":material/scatter_plot:"),
    st.Page("pages/imputation.py", title="Missing data & imputation", icon=":material/healing:"),
    st.Page("pages/clustering.py", title="Clustering", icon=":material/bubble_chart:"),
    st.Page("pages/classification.py", title="Classification & imbalance", icon=":material/rule:"),
]

navigation = st.navigation(pages)
sidebar_settings()
navigation.run()
