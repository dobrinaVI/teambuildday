import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="Data Explorer", layout="wide")

# This is a quick data explorer with an upload page to upload a csv file, a dashboard page
# showing distributions of the uploaded data's numeric columns, and an about page.

# Define each page as a function. This version of dataiku's native streamlit apps
# does not support breaking the code into multiple python files.

def upload_page():
    st.header("📤 Upload data")
    df, file = get_context()

    if df is not None:
        st.write(f"You are currently exploring `{file.name}`")
        st.dataframe(df)
        st.page_link(get_page("Distributions"), label="Vizualize uploaded data", icon="📊")

        if st.button("Upload New File"):
            clear_context()
            st.rerun()  # refresh upload page

    else:
        file = st.file_uploader("Upload a CSV file to explore", type="csv")

        if file:
            df = pd.read_csv(file)
            set_context(df, file)
            st.dataframe(df)
            st.page_link(get_page("Distributions"), label="Vizualize uploaded data", icon="📊")


def dashboard_page():
    st.header("📊 Distributions")

    df, file = get_context()

    if df is None:
        st.warning("No data found! Please upload data first.")
        st.page_link(get_page("Upload data"), label="Upload data", icon="📤")
        return

    # Automatically detect numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 10)

        st.subheader(f"Distribution of `{selected_col}` in `{file.name}`")
        hist_data = pd.Series(df[selected_col].dropna())
        counts, bin_edges = np.histogram(hist_data, bins=bins)
        hist_df = pd.DataFrame({"bin_start": bin_edges[:-1], "count": counts})
        st.bar_chart(hist_df.set_index("bin_start"))

        # Optional summary statistics
        st.markdown("**Summary:**")
        st.dataframe(hist_data.describe().to_frame().T)
    else:
        st.warning("No numeric columns detected in the uploaded dataset.")


def about_page():
    st.header("ℹ️ About")
    st.markdown("""
    This example demonstrates a single-file multipage layout using:
    - `st.Page(...)` to define pages
    - `st.navigation(...)` to setup navigation
    - `st.session_state` to share context across pages
    - `st.page_link(...)` to create custom navigation links 
    """)
    st.info("Requires streamlit 1.37 or later")


# Navigation
pages = [
    st.Page(upload_page, title="Upload data", icon="📤"),
    st.Page(dashboard_page, title="Distributions", icon="📊"),
    st.Page(about_page, title="About", icon="ℹ️"),
]
nav = st.navigation(pages, position="sidebar", expanded=True)


# Utilities
def get_page(title):
    return next(p for p in pages if p.title == title)

def get_context():
    return st.session_state.get("uploaded_data", None), st.session_state.get('uploaded_file', None)

def set_context(df, file):
    st.session_state["uploaded_data"] = df
    st.session_state["uploaded_file"] = file

def clear_context():
    st.session_state.pop("uploaded_data", None)
    st.session_state.pop("uploaded_file", None)


# run the app
nav.run()

