import pandas as pd

# -------------------------------
# Load datasets
# -------------------------------
df_market = pd.read_parquet("us_market_dataset.parquet")
df_nasdaq = pd.read_csv("sector_ticker_mapping.csv")  # Your NASDAQ CSV

# Ensure tickers are strings and uppercase
df_market['ticker'] = df_market['ticker'].astype(str).str.upper()
df_nasdaq['Symbol'] = df_nasdaq['Symbol'].astype(str).str.upper()

# -------------------------------
# Merge NASDAQ sector/industry info
# -------------------------------
df_merged = df_market.merge(
    df_nasdaq[['Symbol', 'Sector', 'Industry']],
    left_on='ticker',
    right_on='Symbol',
    how='left'
)

# Drop redundant 'Symbol' column
df_merged = df_merged.drop(columns=['Symbol'])
# Rename sector/industry columns to lowercase
df_merged = df_merged.rename(columns={'Sector': 'sector', 'Industry': 'industry'})

# -------------------------------
# Stats with percentages
# -------------------------------
total_rows = len(df_merged)

for col in ['sector', 'industry']:
    missing = df_merged[col].isna().sum()
    unique_values = df_merged[col].nunique(dropna=True)
    
    print(f"\nColumn: {col}")
    print(f"  Unique values: {unique_values}")
    print(f"  Missing values: {missing} ({missing/total_rows:.2%})")
    
    # Frequency with percentage
    freq = df_merged[col].value_counts(dropna=True)
    freq_pct = df_merged[col].value_counts(normalize=True, dropna=True) * 100
    freq_df = pd.DataFrame({'count': freq, 'percent': freq_pct})
    print(f"  Top 10 most frequent values:\n{freq_df.head(10)}")

# -------------------------------
# List tickers with missing sector
# -------------------------------
missing_sector_tickers = df_merged.loc[df_merged['sector'].isna(), 'ticker'].unique()
print(f"\nTickers with missing sector ({len(missing_sector_tickers)}):")
print(missing_sector_tickers)

print(df_merged.columns)

# -------------------------------
# Save to new Parquet
# -------------------------------
df_merged.to_parquet("us_market_dataset_with_sector.parquet", index=False)
print(df_merged.shape)
df_merged = df_merged.dropna(subset=['sector'])
print(df_merged.shape)
print("Saved new dataset with Sector and Industry!")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

# --------------------------------------
# 1️⃣ Sector Distribution (Counts)
# --------------------------------------

sector_counts = df_merged['sector'].value_counts().sort_values(ascending=False)
sector_pct = df_merged['sector'].value_counts(normalize=True) * 100

plt.figure(figsize=(10,6))
sns.barplot(x=sector_counts.values, y=sector_counts.index)
plt.title("Distribution of Observations per Sector")
plt.xlabel("Number of Observations")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig("sector_distribution_counts.pdf")
plt.close()

# --------------------------------------
# 2️⃣ Sector Distribution (Percent)
# --------------------------------------

plt.figure(figsize=(10,6))
sns.barplot(x=sector_pct.values, y=sector_pct.index)
plt.title("Distribution of Observations per Sector (%)")
plt.xlabel("Percentage of Dataset")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig("sector_distribution_percent.pdf")
plt.close()

# --------------------------------------
# 3️⃣ Top 20 Industries
# --------------------------------------

industry_counts = df_merged['industry'].value_counts().head(20)

plt.figure(figsize=(10,8))
sns.barplot(x=industry_counts.values, y=industry_counts.index)
plt.title("Top 20 Industries by Observation Count")
plt.xlabel("Number of Observations")
plt.ylabel("Industry")
plt.tight_layout()
plt.savefig("top20_industries.pdf")
plt.close()

# --------------------------------------
# 4️⃣ Unique Tickers per Sector
# --------------------------------------

tickers_per_sector = (
    df_merged.groupby('sector')['ticker']
    .nunique()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,6))
sns.barplot(x=tickers_per_sector.values, y=tickers_per_sector.index)
plt.title("Number of Unique Tickers per Sector")
plt.xlabel("Number of Unique Companies")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig("unique_tickers_per_sector.pdf")
plt.close()

# --------------------------------------
# 5️⃣ Sector Concentration (Herfindahl Index)
# --------------------------------------

sector_shares = sector_counts / sector_counts.sum()
herfindahl_index = np.sum(sector_shares ** 2)

print("\nSector Concentration (Herfindahl Index):", round(herfindahl_index, 4))

# Interpretation guide
if herfindahl_index < 0.15:
    print("→ Dataset is well diversified across sectors.")
elif herfindahl_index < 0.25:
    print("→ Moderate concentration.")
else:
    print("→ Highly concentrated dataset.")

# --------------------------------------
# 6️⃣ Export Summary Table for Paper
# --------------------------------------

sector_summary = pd.DataFrame({
    "count": sector_counts,
    "percent": sector_pct.round(2),
    "unique_tickers": tickers_per_sector
})

sector_summary.to_csv("sector_summary_table.csv")
print("\nSaved sector summary table.")

# --------------------------------------
# 7️⃣ Industry Fragmentation Check
# --------------------------------------

industry_size_stats = df_merged['industry'].value_counts().describe()
print("\nIndustry Size Distribution Summary:")
print(industry_size_stats)