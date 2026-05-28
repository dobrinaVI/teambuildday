import numpy as np
import pandas as pd
import dataiku


input_ds = dataiku.Dataset("customer_data")
df = input_ds.get_dataframe()

df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["signup_year"] = df["signup_date"].dt.year.astype("Int64")
df["signup_month"] = df["signup_date"].dt.month.astype("Int64")

df["is_subscription"] = (df["plan"] == "subscription").astype("int64")

for r in ["NA", "EU", "APAC"]:
    df[f"region_{r.lower()}"] = (df["region"] == r).astype("int64")

df["log_total_spend_usd"] = np.log1p(df["total_spend_usd"].astype(float))
df["spend_per_tenure_month"] = df["total_spend_usd"] / np.maximum(1, df["tenure_months"])

df["returns_rate"] = df["returned_items_last_year"] / (1.0 + (df["total_spend_usd"] / 250.0))
df["tickets_per_return"] = df["support_tickets_last_year"] / np.maximum(1, df["returned_items_last_year"])

df["low_open_rate"] = (df["marketing_email_open_rate"] < 0.12).astype("int64")
df["high_returns"] = (df["returned_items_last_year"] >= 3).astype("int64")

df["recency_bucket"] = pd.cut(
    df["last_purchase_days_ago"],
    bins=[-1, 30, 90, 180, 365],
    labels=["0-30", "31-90", "91-180", "181-365"],
).astype("string")

df["tenure_bucket"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 2, 6, 12, 24, 60, 10_000],
    labels=["0-2", "3-6", "7-12", "13-24", "25-60", "60+"],
).astype("string")

df["age_bucket"] = pd.cut(
    df["age"],
    bins=[15, 24, 34, 44, 54, 80],
    labels=["16-24", "25-34", "35-44", "45-54", "55-80"],
).astype("string")

output_ds = dataiku.Dataset("customer_features")
output_ds.write_with_schema(df)

