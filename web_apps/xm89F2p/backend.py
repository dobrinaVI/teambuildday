import json

import pandas as pd
import streamlit as st

try:
    import dataiku  # available in DSS
except Exception:  # pragma: no cover
    dataiku = None


st.set_page_config(page_title="Churn Risk Explorer", layout="wide")

st.title("Churn Risk Explorer")
st.caption("Predict churn risk and explain top drivers; includes a tenure=12 months what-if.")

DEFAULT_SCORED_DS = "customer_churn_scored"
DEFAULT_WHATIF_DS = "customer_churn_scored_tenure12"


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataiku is None:
        raise RuntimeError("This app must run inside Dataiku DSS (missing 'dataiku' module).")
    return dataiku.Dataset(dataset_name).get_dataframe()


@st.cache_data(show_spinner=False)
def load_scored(scored_ds: str, whatif_ds: str):
    scored = load_dataset(scored_ds)
    whatif = load_dataset(whatif_ds)
    return scored, whatif


def parse_explanations(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def top_k_explanations(expl: dict, k: int = 3):
    items = [(str(feature), float(val)) for feature, val in expl.items()]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:k]


with st.sidebar:
    st.header("Inputs")
    scored_ds = st.text_input("Scored dataset", value=DEFAULT_SCORED_DS)
    whatif_ds = st.text_input("What-if (tenure=12) scored dataset", value=DEFAULT_WHATIF_DS)
    st.divider()
    st.caption("Tip: Ensure your scoring recipe outputs the 'explanations' column.")

scored_df, whatif_df = load_scored(scored_ds, whatif_ds)

ID_COL = "customer_id"
RISK_COL = "proba_1"

if ID_COL not in scored_df.columns or RISK_COL not in scored_df.columns:
    st.error(
        f"Expected columns '{ID_COL}' and '{RISK_COL}' in {scored_ds}. "
        f"Found: {list(scored_df.columns)}"
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    regions = sorted(scored_df.get("region", pd.Series(dtype=str)).dropna().unique().tolist())
    region_filter = st.multiselect("Region", regions)
with col2:
    plans = sorted(scored_df.get("plan", pd.Series(dtype=str)).dropna().unique().tolist())
    plan_filter = st.multiselect("Plan", plans)
with col3:
    min_risk = st.slider("Min churn risk", 0.0, 1.0, 0.0, 0.01)
with col4:
    max_rows = st.number_input("Max rows", min_value=50, max_value=10000, value=500, step=50)

filtered = scored_df.copy()
if region_filter and "region" in filtered.columns:
    filtered = filtered[filtered["region"].isin(region_filter)]
if plan_filter and "plan" in filtered.columns:
    filtered = filtered[filtered["plan"].isin(plan_filter)]
filtered = filtered[filtered[RISK_COL].astype(float) >= float(min_risk)]

filtered = filtered.sort_values(RISK_COL, ascending=False)

st.subheader("Prioritize outreach")
show_cols = [
    c
    for c in [
        ID_COL,
        RISK_COL,
        "region",
        "plan",
        "tenure_months",
        "returned_items_last_year",
        "marketing_email_open_rate",
        "last_purchase_days_ago",
    ]
    if c in filtered.columns
]
st.dataframe(filtered[show_cols].head(int(max_rows)), use_container_width=True)

ids = filtered[ID_COL].astype(int).tolist()
selected_id = st.selectbox("Select a customer_id", ids)

row = scored_df.loc[scored_df[ID_COL].astype(int) == int(selected_id)].iloc[0]
risk = float(row[RISK_COL])

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Churn risk")
    st.metric("P(churn in 30d)", f"{risk:.2f}")

    st.subheader("Top 3 influences")
    expl = parse_explanations(row.get("explanations"))
    top3 = top_k_explanations(expl, 3)
    if not top3:
        st.info("No per-row explanations found. Ensure scoring recipe outputs explanations.")
    else:
        table = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "influence": round(val, 2),
                    "direction": "towards churn" if val > 0 else "towards no churn",
                }
                for feature, val in top3
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

with right:
    st.subheader("What-if: tenure = 12 months")
    if ID_COL not in whatif_df.columns or RISK_COL not in whatif_df.columns:
        st.info(f"What-if dataset {whatif_ds} missing '{ID_COL}'/'{RISK_COL}'.")
    else:
        match = whatif_df.loc[whatif_df[ID_COL].astype(int) == int(selected_id)]
        if len(match) == 0:
            st.info("Customer not found in what-if dataset.")
        else:
            whatif_risk = float(match.iloc[0][RISK_COL])
            current_tenure = row.get("tenure_months")
            st.write(
                f"If this customer's tenure were **12** months instead of **{int(current_tenure) if pd.notna(current_tenure) else 'N/A'}**, "
                f"the churn risk would be **{whatif_risk:.2f}** (was **{risk:.2f}**)."
            )

st.subheader("Customer details")
detail = row.to_dict()
detail.pop("explanations", None)
st.json(detail)
