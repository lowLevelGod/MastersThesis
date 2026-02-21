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