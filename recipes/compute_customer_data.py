import numpy as np
import pandas as pd
import dataiku

# =========
# Settings
# =========
N = 10_000
rng = np.random.default_rng(42)
today = pd.Timestamp.today().normalize()

# =================
# Helper functions
# =================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def annual_to_p30(annual_rate):
    return 1 - (1 - annual_rate) ** (30 / 365)

def calibrate_intercept(logits_wo_intercept, target_mean, iters=60):
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = sigmoid(logits_wo_intercept + mid).mean()
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ==========
# Generate
# ==========
customer_id = np.arange(1, N + 1)

region = rng.choice(["NA", "EU", "APAC"], size=N, p=[0.45, 0.35, 0.20])
plan = rng.choice(["subscription", "one-time"], size=N, p=[0.55, 0.45])

# Signup date within last 5 years
days_back = rng.integers(0, 5 * 365 + 1, size=N)
signup_date = (today - pd.to_timedelta(days_back, unit="D")).normalize()

tenure_months = np.maximum(
    0,
    ((today.year - signup_date.year) * 12 + (today.month - signup_date.month)).astype(int),
)

age = np.clip(np.rint(rng.normal(38, 12, size=N)).astype(int), 16, 80)

# Marketing email open rate (0..1), influenced by plan/region/age
age_norm = (age - 38) / 12
region_effect = np.select(
    [region == "NA", region == "EU", region == "APAC"],
    [0.03, 0.05, -0.02],
    default=0.0,
)
plan_effect_open = np.where(plan == "subscription", 0.05, -0.02)
mean_open = 0.28 + 0.04 * (-age_norm) + region_effect + plan_effect_open
mean_open = np.clip(mean_open, 0.03, 0.75)

k = 18.0
alpha = np.clip(mean_open * k, 0.6, None)
beta = np.clip((1 - mean_open) * k, 0.6, None)
marketing_email_open_rate = np.clip(rng.beta(alpha, beta), 0.0, 1.0)

# Total spend (heavy tail)
region_spend_mult = np.select(
    [region == "NA", region == "EU", region == "APAC"],
    [1.10, 1.00, 0.85],
    default=1.0,
)
plan_spend_mult = np.where(plan == "subscription", 1.20, 0.95)
tenure_factor = np.log1p(tenure_months) / np.log1p(60)

mu = np.log(220) + 0.9 * tenure_factor + 0.15 * (age - 35) / 20
sigma = 0.75
total_spend_usd = np.exp(rng.normal(mu, sigma, size=N)) * region_spend_mult * plan_spend_mult
total_spend_usd = np.round(np.clip(total_spend_usd, 0, None), 2)

# Recency (days ago): worse with low open rate; one-time tends to be older
open_influence = (0.55 - marketing_email_open_rate)
base_recency = np.where(
    plan == "subscription",
    rng.gamma(2.2, 12.0, size=N),
    rng.gamma(2.0, 22.0, size=N),
)
last_purchase_days_ago = base_recency + 18.0 * open_influence + rng.normal(0, 4, size=N)
last_purchase_days_ago = np.clip(np.rint(last_purchase_days_ago), 0, 365).astype(int)

# Returns: increase with spend + dissatisfaction (low open)
spend_scale = np.log1p(total_spend_usd) / np.log1p(np.percentile(total_spend_usd, 95))
returns_lambda = 0.15 + 1.25 * spend_scale + 0.9 * np.clip(open_influence, 0, 1)
returned_items_last_year = np.clip(rng.poisson(returns_lambda), 0, 25).astype(int)

# Support tickets: tied to returns + recency
tickets_lambda = 0.10 + 0.20 * returned_items_last_year + 0.004 * last_purchase_days_ago
support_tickets_last_year = np.clip(rng.poisson(tickets_lambda), 0, 30).astype(int)

# =========
# Churned (30d) with plan-specific base rates
# =========
p30_sub = annual_to_p30(0.18)
p30_one = annual_to_p30(0.35)

score = (
    1.05 * (marketing_email_open_rate < 0.12).astype(float)
    + 0.55 * np.clip(0.18 - marketing_email_open_rate, 0, 0.18) / 0.18
    + 0.25 * np.clip(last_purchase_days_ago - 60, 0, 305) / 305
    + 0.30 * np.clip(returned_items_last_year - 2, 0, 23) / 23
    + 0.22 * np.clip(support_tickets_last_year - 1, 0, 29) / 29
    + 0.10 * (tenure_months < 3).astype(float)
)
logits_wo = score - score.mean()

churn_prob = np.empty(N, dtype=float)
mask_sub = plan == "subscription"
mask_one = ~mask_sub

delta_sub = calibrate_intercept(logits_wo[mask_sub], p30_sub)
delta_one = calibrate_intercept(logits_wo[mask_one], p30_one)

churn_prob[mask_sub] = sigmoid(logits_wo[mask_sub] + delta_sub)
churn_prob[mask_one] = sigmoid(logits_wo[mask_one] + delta_one)

churned_30d = rng.binomial(1, churn_prob, size=N).astype(int)

# =========
# Output
# =========
df = pd.DataFrame(
    {
        "customer_id": customer_id,
        "signup_date": signup_date,  # keep as date/datetime
        "region": region,
        "plan": plan,
        "age": age,
        "tenure_months": tenure_months,
        "total_spend_usd": total_spend_usd,
        "last_purchase_days_ago": last_purchase_days_ago,
        "support_tickets_last_year": support_tickets_last_year,
        "returned_items_last_year": returned_items_last_year,
        "marketing_email_open_rate": np.round(marketing_email_open_rate, 4),
        "churned_30d": churned_30d,
    }
)

output_ds = dataiku.Dataset("customer_data")
output_ds.write_with_schema(df)
