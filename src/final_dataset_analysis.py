import pandas as pd

# Load output parquet
df = pd.read_parquet("sec_filings_with_price_features_and_labels.parquet")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Quick peek
print("\nHead:")
print(df.head(5))

# Random samples (good for sanity check)
print("\nRandom samples:")
print(df.sample(5, random_state=42)[
    ["ticker", "filing_date", "acceptance_time", "event_day",
     "close_last", "ma20", "volatility_20",
     "return_1d", "label_1d", "return_5d", "label_5d"]
])

# Check label distribution for 1-day horizon
print("\nLabel distribution (1d):")
print(df["label_1d"].value_counts(dropna=False))

# Basic return stats
print("\nReturn stats (1d):")
print(df["return_1d"].describe())