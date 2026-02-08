import pandas as pd
from tqdm import tqdm

class SecFilingsOutlierRemover:
    """
    This script performs outlier removal on the parsed SEC filings dataset.
    It drops rows where both MDA and Market Risk sections are empty, and then
    removes extreme outliers based on text length percentiles.
    """
    MDA_COL = "mda_text"
    MR_COL = "market_risk_text"

    def remove_outliers(self, 
                        input_parquet: str,
                        output_parquet: str,
                        percentile_cutoff: float):
        
        print("Loading parquet...")
        df = pd.read_parquet(input_parquet)

        print(f"Loaded rows: {len(df):,}")
        
        print("\nDropping rows where both MDA and Market Risk are missing...")
        def is_empty_text(series: pd.Series) -> pd.Series:
            """Return boolean mask where text is null or empty/whitespace."""
            return series.isna() | (series.astype(str).str.strip() == "")

        mda_empty = is_empty_text(df[self.MDA_COL])
        mr_empty = is_empty_text(df[self.MR_COL])

        before = len(df)
        # Drop rows where BOTH mda_text and market_risk_text are empty
        df = df[~(mda_empty & mr_empty)].copy()
        after = len(df)

        print(f"Dropped {before - after:,} rows (both empty). Remaining: {after:,}")

        print(f"\nComputing {int(percentile_cutoff*100)}th percentile cutoffs...")

        tqdm.pandas(desc="Computing MDA lengths")
        df["mda_len"] = df[self.MDA_COL].fillna("").astype(str).progress_apply(len)

        tqdm.pandas(desc="Computing Market Risk lengths")
        df["market_risk_len"] = df[self.MR_COL].fillna("").astype(str).progress_apply(len)

        # Remove extreme outliers by percentile cutoff 
        mda_cutoff = df["mda_len"].quantile(percentile_cutoff)
        mr_cutoff = df["market_risk_len"].quantile(percentile_cutoff)

        print(f"MDA length cutoff ({percentile_cutoff:.2f} quantile): {int(mda_cutoff):,}")
        print(f"Market Risk length cutoff ({percentile_cutoff:.2f} quantile): {int(mr_cutoff):,}")

        print("\nFiltering outliers...")

        before = len(df)

        df = df[(df["mda_len"] <= mda_cutoff) | (df["mda_len"] == 0)]

        mid = len(df)
        print(f"After MDA outlier removal: {mid:,} rows remaining")

        df = df[(df["market_risk_len"] <= mr_cutoff) | (df["market_risk_len"] == 0)]

        after = len(df)
        print(f"After Market Risk outlier removal: {after:,} rows remaining")
        print(f"Total removed by outlier filtering: {before - after:,}")

        # Drop helper columns
        df = df.drop(columns=["mda_len", "market_risk_len"], errors="ignore")

        print(f"\nSaving cleaned parquet to: {output_parquet}")
        df.to_parquet(output_parquet, index=False)

        print("Done.")
        print(f"Final dataset size: {len(df):,}")


if __name__ == "__main__":
    secFilingsOutlierRemover = SecFilingsOutlierRemover()
    
    secFilingsOutlierRemover.remove_outliers(
        input_parquet = "parsed_sec_filings.parquet",
        output_parquet = "parsed_sec_filings_cleaned.parquet",
        percentile_cutoff = 0.99
    )