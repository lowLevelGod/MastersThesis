import pandas as pd
import numpy as np

PARQUET_PATH = "sec_filings_with_price_features_and_labels.parquet"
N_SAMPLES = 10

df = pd.read_parquet(PARQUET_PATH)

print("=" * 120)
print("DATASET OVERVIEW")
print("=" * 120)
print("Shape:", df.shape)
print("Memory usage (MB):", df.memory_usage(deep=True).sum() / 1e6)
print("\nDtypes:\n", df.dtypes)

print("\n" + "=" * 120)
print("BASIC HEAD / TAIL")
print("=" * 120)
print("\nHead:\n", df.head(3))
print("\nTail:\n", df.tail(3))

# --------------------------------------------------------------------------------
# Missing values check (NaNs + empty strings)
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("MISSING VALUES CHECK")
print("=" * 120)

missing_summary = []

for col in df.columns:
    series = df[col]

    nan_count = series.isna().sum()

    # count empty strings only if object/string column
    empty_count = 0
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        empty_count = (series.fillna("").astype(str).str.strip() == "").sum()

    total_missing = nan_count + empty_count
    missing_rate = total_missing / len(df)

    missing_summary.append((col, nan_count, empty_count, total_missing, missing_rate))

missing_df = pd.DataFrame(
    missing_summary,
    columns=["column", "nan_count", "empty_string_count", "total_missing", "missing_rate"]
).sort_values("missing_rate", ascending=False)

print(missing_df.head(40).to_string(index=False))

# --------------------------------------------------------------------------------
# Duplicate key checks
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("DUPLICATE CHECKS")
print("=" * 120)

possible_key_cols = [c for c in ["accession_number", "ticker", "filing_date"] if c in df.columns]
if possible_key_cols:
    dup_count = df.duplicated(subset=possible_key_cols).sum()
    print(f"Duplicate rows based on {possible_key_cols}: {dup_count:,}")
else:
    print("No standard key columns found for duplicate check.")

# --------------------------------------------------------------------------------
# Basic ticker sanity checks
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("TICKER SANITY CHECK")
print("=" * 120)

if "ticker" in df.columns:
    print("Unique tickers:", df["ticker"].nunique(dropna=True))
    print("Top 20 tickers:\n", df["ticker"].value_counts(dropna=False).head(20))

    bad_tickers = df["ticker"].dropna().astype(str)
    weird = bad_tickers[bad_tickers.str.len() > 10]
    print("Tickers with length > 10:", len(weird))

    none_like = df["ticker"].astype(str).str.lower().isin(["none", "nan", "null", ""])
    print("Tickers that look like None/null/empty:", none_like.sum())

# --------------------------------------------------------------------------------
# Date/time checks
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("DATE/TIME SANITY CHECK")
print("=" * 120)

if "filing_date" in df.columns:
    filing_date = pd.to_datetime(df["filing_date"], errors="coerce")
    bad_dates = filing_date.isna().sum()
    print("Invalid filing_date values:", bad_dates)
    print("Min filing_date:", filing_date.min())
    print("Max filing_date:", filing_date.max())

if "acceptance_time" in df.columns:
    # acceptance_time might be "HH:MM" or full timestamp; just check coercion
    acc = pd.to_datetime(df["acceptance_time"], errors="coerce")
    print("Invalid acceptance_time values:", acc.isna().sum())

# --------------------------------------------------------------------------------
# Numeric feature checks
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("NUMERIC COLUMN CHECKS")
print("=" * 120)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", len(numeric_cols))

num_summary = []
for col in numeric_cols:
    s = df[col]
    num_summary.append({
        "col": col,
        "nan_count": int(s.isna().sum()),
        "inf_count": int(np.isinf(s).sum()) if np.issubdtype(s.dtype, np.number) else 0,
        "min": s.min(skipna=True),
        "max": s.max(skipna=True),
        "mean": s.mean(skipna=True),
        "std": s.std(skipna=True),
        "zeros": int((s == 0).sum()) if np.issubdtype(s.dtype, np.number) else 0
    })

num_df = pd.DataFrame(num_summary).sort_values("nan_count", ascending=False)
print(num_df.head(40).to_string(index=False))

# --------------------------------------------------------------------------------
# Label checks
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("LABEL CHECKS")
print("=" * 120)

label_cols = [c for c in df.columns if c.startswith("label_")]
return_cols = [c for c in df.columns if c.startswith("return_")]

print("Label columns found:", label_cols)
print("Return columns found:", return_cols)

for col in label_cols:
    print("\n" + "-" * 80)
    print(f"Label distribution for {col}:")
    print(df[col].value_counts(dropna=False))

# Check if returns and labels mismatch
for rcol in return_cols:
    if "label" in rcol:
        continue
    horizon = rcol.replace("return_", "")
    lcol = f"label_{horizon}"
    if lcol in df.columns:
        bad = df[rcol].notna() & df[lcol].isna()
        print(f"Return present but label missing for horizon {horizon}: {bad.sum():,}")

# --------------------------------------------------------------------------------
# Check for impossible price relationships (if columns exist)
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("PRICE CONSISTENCY CHECKS")
print("=" * 120)

price_cols = ["open", "high", "low", "close"]
existing_price_cols = [c for c in df.columns if any(c.endswith(x) for x in price_cols)]

# Common ones might be: close_last, open_next, close_next, etc.
close_like = [c for c in df.columns if "close" in c.lower()]
open_like = [c for c in df.columns if "open" in c.lower()]
high_like = [c for c in df.columns if "high" in c.lower()]
low_like  = [c for c in df.columns if "low" in c.lower()]

def check_price_bounds(high_col, low_col, close_col):
    if high_col in df.columns and low_col in df.columns and close_col in df.columns:
        bad = (df[close_col] > df[high_col]) | (df[close_col] < df[low_col])
        print(f"{close_col} outside [{low_col},{high_col}] count:", bad.sum())

# Try some common combos if they exist
for close_col in close_like:
    for high_col in high_like:
        for low_col in low_like:
            # only check if suffixes match roughly
            if close_col.split("_")[-1] == high_col.split("_")[-1] == low_col.split("_")[-1]:
                check_price_bounds(high_col, low_col, close_col)

# --------------------------------------------------------------------------------
# Text column quick sanity check (length distributions)
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("TEXT COLUMN SANITY CHECK")
print("=" * 120)

text_cols = [c for c in ["mda_text", "market_risk_text"] if c in df.columns]
for col in text_cols:
    lens = df[col].fillna("").astype(str).str.len()
    nonzero = lens[lens > 0]
    print(f"\n--- {col} ---")
    print("Missing/empty:", (lens == 0).sum())
    print("Non-empty count:", len(nonzero))
    print(nonzero.describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

# --------------------------------------------------------------------------------
# Random sample inspection (good for catching subtle alignment bugs)
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print(f"RANDOM SAMPLE INSPECTION ({N_SAMPLES} rows)")
print("=" * 120)

sample_cols = [
    c for c in [
        "ticker", "filing_date", "acceptance_time", "event_day",
        "close_last", "open_next", "close_next",
        "ma5", "ma20", "volatility_20", "volume_zscore_20",
        "return_1d", "label_1d", "return_5d", "label_5d"
    ]
    if c in df.columns
]

sample_df = df.sample(min(N_SAMPLES, len(df)), random_state=42)
print(sample_df[sample_cols].to_string(index=False))

# --------------------------------------------------------------------------------
# Check that returns roughly match the price columns (if present)
# (example: return_1d should be close_next / close_last - 1)
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("RETURN CONSISTENCY CHECK")
print("=" * 120)

if all(c in df.columns for c in ["close_last", "close_next", "return_1d"]):
    calc = (df["close_next"] / df["close_last"]) - 1
    diff = (df["return_1d"] - calc).abs()

    print("Mean abs diff return_1d vs computed:", diff.mean(skipna=True))
    print("Max abs diff return_1d vs computed:", diff.max(skipna=True))

    # show worst offenders
    worst = diff.sort_values(ascending=False).head(10).index
    print("\nWorst 10 return mismatches:")
    print(df.loc[worst, ["ticker", "filing_date", "close_last", "close_next", "return_1d"]])

# --------------------------------------------------------------------------------
# Final quick summary
# --------------------------------------------------------------------------------
print("\n" + "=" * 120)
print("FINAL SUMMARY")
print("=" * 120)
print("Total rows:", len(df))
print("Total columns:", df.shape[1])
print("Rows with any NaN:", df.isna().any(axis=1).sum())
print("Rows with all numeric features NaN:", df[numeric_cols].isna().all(axis=1).sum() if numeric_cols else "N/A")
print("Done.")