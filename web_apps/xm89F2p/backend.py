import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Explorer", layout="wide")

st.title("Churn Risk Explorer")
st.caption("Predict churn risk and explain top drivers (no API calls).")

EXPECTED_COLUMNS = [
    "customer_id",
    "signup_date",
    "signup_month",
    "signup_year",
    "region",
    "region_apac",
    "region_eu",
    "region_na",
    "plan",
    "is_subscription",
    "age",
    "age_bucket",
    "tenure_months",
    "tenure_bucket",
    "total_spend_usd",
    "spend_per_tenure_month",
    "log_total_spend_usd",
    "last_purchase_days_ago",
    "recency_bucket",
    "support_tickets_last_year",
    "returned_items_last_year",
    "high_returns",
    "tickets_per_return",
    "returns_rate",
    "marketing_email_open_rate",
    "low_open_rate",
    "churned_30d",
    "proba_0",
    "proba_1",
    "prediction",
    "explanations",
]


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


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file, separator: str, has_header: bool) -> pd.DataFrame:
    header = 0 if has_header else None
    df = pd.read_csv(uploaded_file, sep=separator, header=header, compression="infer")
    if not has_header and df.shape[1] == len(EXPECTED_COLUMNS):
        df.columns = EXPECTED_COLUMNS
    return df


def example_dataset(n: int = 1000) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        proba = (i % 100) / 100
        expl = {
            "returned_items_last_year": (proba - 0.5) * 0.9,
            "marketing_email_open_rate": (0.5 - proba) * 0.7,
            "last_purchase_days_ago": (proba - 0.4) * 0.5,
        }
        rows.append(
            {
                "customer_id": i,
                "proba_1": proba,
                "region": ["NA", "EU", "APAC"][i % 3],
                "plan": ["subscription", "one-time"][i % 2],
                "tenure_months": (i % 60) + 1,
                "explanations": json.dumps(expl),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload scored file", type=["csv", "tsv", "gz"])
    separator = st.text_input("Separator", value="\\t", help="Use \\t for tab-separated files.")
    has_header = st.toggle("File has header row", value=True)
    st.caption(
        "Expected columns include `customer_id`, `proba_1`, and `explanations` (JSON per row). "
        "If your file has no header, this app auto-assigns known column names when the column count matches."
    )
    use_example = st.toggle("Use example data", value=uploaded is None)

if use_example:
    scored_df = example_dataset(1000)
else:
    if uploaded is None:
        st.info("Upload a scored CSV to begin, or enable example data.")
        st.stop()
    scored_df = load_csv(uploaded, separator=separator.encode("utf-8").decode("unicode_escape"), has_header=has_header)

with st.sidebar:
    st.divider()
    st.header("Columns")
    ID_COL = st.text_input("ID column", value="customer_id")
    RISK_COL = st.text_input("Risk column", value="proba_1")
    EXPL_COL = st.text_input("Explanations column", value="explanations")

missing = [c for c in [ID_COL, RISK_COL] if c not in scored_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Found: {list(scored_df.columns)}")
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

ids = filtered[ID_COL].dropna().astype(int).tolist()
if len(ids) == 0:
    st.info("No rows match the current filters.")
    st.stop()

selected_id = st.selectbox("Select a customer_id", ids, index=0)

row = scored_df.loc[scored_df[ID_COL].astype(int) == int(selected_id)].iloc[0]
risk = float(row[RISK_COL])

st.divider()
st.subheader("Churn risk")
st.metric("P(churn in 30d)", f"{risk:.2f}")

st.subheader("Top 3 influences")
expl = parse_explanations(row.get(EXPL_COL))
top3 = top_k_explanations(expl, 3)
if not top3:
    if EXPL_COL not in scored_df.columns:
        st.info("No explanations column found. Add an `explanations` column (JSON per row) to show drivers.")
    else:
        st.info("No per-row explanations found for this customer.")
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

st.subheader("Customer details")
detail = row.to_dict()
detail.pop(EXPL_COL, None)
st.json(detail)
