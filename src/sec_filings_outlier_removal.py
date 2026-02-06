import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_PARQUET = "parsed_sec_filings.parquet"
OUTPUT_PARQUET = "parsed_sec_filings_cleaned.parquet"

MDA_COL = "mda_text"
MR_COL = "market_risk_text"

PERCENTILE_CUTOFF = 0.99  # change to 0.99 if you want less aggressive filtering


def is_empty_text(series: pd.Series) -> pd.Series:
    """Return boolean mask where text is null or empty/whitespace."""
    return series.isna() | (series.astype(str).str.strip() == "")


def main():
    print("Loading parquet...")
    df = pd.read_parquet(INPUT_PARQUET)

    print(f"Loaded rows: {len(df):,}")

    # ------------------------------------------------------------------
    # Step 1: Drop rows where BOTH mda_text and market_risk_text are empty
    # ------------------------------------------------------------------
    print("\nDropping rows where both MDA and Market Risk are missing...")

    mda_empty = is_empty_text(df[MDA_COL])
    mr_empty = is_empty_text(df[MR_COL])

    before = len(df)
    df = df[~(mda_empty & mr_empty)].copy()
    after = len(df)

    print(f"Dropped {before - after:,} rows (both empty). Remaining: {after:,}")

    # ------------------------------------------------------------------
    # Step 2: Remove extreme outliers by percentile cutoff (MDA first)
    # ------------------------------------------------------------------
    print(f"\nComputing {int(PERCENTILE_CUTOFF*100)}th percentile cutoffs...")

    tqdm.pandas(desc="Computing MDA lengths")
    df["mda_len"] = df[MDA_COL].fillna("").astype(str).progress_apply(len)

    tqdm.pandas(desc="Computing Market Risk lengths")
    df["market_risk_len"] = df[MR_COL].fillna("").astype(str).progress_apply(len)

    mda_cutoff = df["mda_len"].quantile(PERCENTILE_CUTOFF)
    mr_cutoff = df["market_risk_len"].quantile(PERCENTILE_CUTOFF)

    print(f"MDA length cutoff ({PERCENTILE_CUTOFF:.2f} quantile): {int(mda_cutoff):,}")
    print(f"Market Risk length cutoff ({PERCENTILE_CUTOFF:.2f} quantile): {int(mr_cutoff):,}")

    # ------------------------------------------------------------------
    # Apply filtering
    # ------------------------------------------------------------------
    print("\nFiltering outliers...")

    before = len(df)

    # Filter MDA outliers (but keep empty ones)
    df = df[(df["mda_len"] <= mda_cutoff) | (df["mda_len"] == 0)]

    mid = len(df)
    print(f"After MDA outlier removal: {mid:,} rows remaining")

    # Filter Market Risk outliers (but keep empty ones)
    df = df[(df["market_risk_len"] <= mr_cutoff) | (df["market_risk_len"] == 0)]

    after = len(df)
    print(f"After Market Risk outlier removal: {after:,} rows remaining")
    print(f"Total removed by outlier filtering: {before - after:,}")

    # ------------------------------------------------------------------
    # Drop helper columns
    # ------------------------------------------------------------------
    df = df.drop(columns=["mda_len", "market_risk_len"], errors="ignore")

    # ------------------------------------------------------------------
    # Save cleaned parquet
    # ------------------------------------------------------------------
    print(f"\nSaving cleaned parquet to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done.")
    print(f"Final dataset size: {len(df):,}")


if __name__ == "__main__":
    main()