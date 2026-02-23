import pandas as pd
import numpy as np

df = pd.read_parquet("us_market_dataset_with_sector.parquet")
df["event_day"] = pd.to_datetime(df["event_day"])

horizons = [1,2,3,4,5,6,7]

# --------------------------------------------
# LEAVE-ONE-OUT WITH SAFE FALLBACK
# --------------------------------------------

for h in horizons:
    
    col = f"return_{h}d"
    
    # ---------- SECTOR ----------
    sector_group = df.groupby(["event_day", "sector"])[col]
    
    sector_sum = sector_group.transform("sum")
    sector_count = sector_group.transform("count")
    
    # Safe LOO mean
    sector_loo = np.where(
        sector_count > 1,
        (sector_sum - df[col]) / (sector_count - 1),
        df[col]   # fallback when only 1 firm
    )
    
    df[f"return_{h}d_sector_adj"] = df[col] - sector_loo
    
    
    # ---------- INDUSTRY ----------
    industry_group = df.groupby(["event_day", "industry"])[col]
    
    industry_sum = industry_group.transform("sum")
    industry_count = industry_group.transform("count")
    
    industry_loo = np.where(
        industry_count > 1,
        (industry_sum - df[col]) / (industry_count - 1),
        df[col]
    )
    
    df[f"return_{h}d_industry_adj"] = df[col] - industry_loo


# --------------------------------------------
# CLASSIFICATION LABELS (1% threshold)
# --------------------------------------------

def create_label(x, threshold=0.01):
    if x > threshold:
        return 2
    elif x < -threshold:
        return 0
    else:
        return 1


for h in horizons:
    
    df[f"label_{h}d_sector_adj"] = (
        df[f"return_{h}d_sector_adj"]
        .apply(create_label)
    )
    
    df[f"label_{h}d_industry_adj"] = (
        df[f"return_{h}d_industry_adj"]
        .apply(create_label)
    )


# --------------------------------------------
# SAVE
# --------------------------------------------

df.to_parquet(
    "us_market_dataset_sector_and_industry_adjusted_returns.parquet",
    index=False
)

print("Dataset created.")
print(df.columns)
print("Total NaNs per column:")
print(df.isna().sum().sort_values())