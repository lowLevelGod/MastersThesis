import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import textwrap

# PARQUET_PATH = "parsed_sec_filings.parquet"
# N_SAMPLES = 30
# PREVIEW_CHARS = 2500

# TEXT_COLS = ["mda_text", "market_risk_text"]
# META_COLS = ["filing_date", "form_type"]
# cols_to_read = list(dict.fromkeys(TEXT_COLS + META_COLS))


# def pretty_print_sample(row, col, idx):
#     form_type = row.get("form_type")
#     filing_date = row.get("filing_date")
#     text = str(row.get(col, "") if row.get(col) is not None else "")

#     print("\n" + "-" * 120)
#     print(f"[Sample {idx}]")
#     print(f"Form: {form_type} | Date: {filing_date}")
#     print(f"Length: {len(text):,}")

#     if not text.strip():
#         print("\n[EMPTY TEXT]\n")
#         return

#     preview = text

#     print("\n--- PREVIEW ---\n")
#     print(textwrap.fill(preview, width=120))


# # Load full dataframe (EXPENSIVE but true random sampling)
# print("Loading parquet into memory...")
# df = pd.read_parquet(PARQUET_PATH, columns=cols_to_read)

# print(f"Loaded rows: {len(df):,}")
# print(df[["form_type", "filing_date"]].head())

# # Convert date column (optional)
# df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

# # ---------------------------------------------------------
# # Exact random sampling independently for each text column
# # ---------------------------------------------------------
# for col in TEXT_COLS:
#     print("\n" + "=" * 120)
#     print(f"EXACT RANDOM GLOBAL INSPECTION FOR COLUMN: {col}")
#     print("=" * 120)

#     df_col = df.copy()

#     # length stats
#     lengths = df_col[col].fillna("").astype(str).map(len)

#     empty_count = (lengths == 0).sum()

#     print(f"Total rows: {len(df_col):,}")
#     print(f"Empty rows: {empty_count:,} ({empty_count / len(df_col):.3%})")
#     print(f"Length stats: min={lengths.min():,}, mean={lengths.mean():,.0f}, max={lengths.max():,}")

#     # Exact random sample
#     df_sample = df_col.sample(n=N_SAMPLES, random_state=None).reset_index(drop=True)

#     for i, row in df_sample.iterrows():
#         pretty_print_sample(row, col, i)


# # -----------------------------
# # Main quality analysis function
# # -----------------------------
def analyze_extraction_quality(
    sec_df: pd.DataFrame,
    text_cols=("mda_text", "market_risk_text"),
):
    """
    Analyze extracted SEC text sections and detect suspicious extractions.
    Prints summary stats + plots + suspicious examples.
    """

    df = sec_df.copy()

    print("\n==================== BASIC DATASET INFO ====================")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    print("\n==================== MISSING RATE ====================")
    for col in text_cols:
        if col not in df.columns:
            print(f"{col}: column not found")
            continue

        missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        missing_rate = missing_mask.mean()

        print(f"{col}: missing rate = {missing_rate:.3f}")
        
        print("\n==================== LENGTH STATS ====================")
        for col in text_cols:
            if col not in df.columns:
                print(f"{col}: column not found")
                continue

            df[f"{col}_len"] = df[col].fillna("").astype(str).str.len()

            # exclude zero lengths
            lengths = df.loc[df[f"{col}_len"] > 0, f"{col}_len"]

            if lengths.empty:
                print(f"\n--- {col} length stats ---")
                print("No non-empty values.")
                continue

            stats = lengths.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            print(f"\n--- {col} length stats (excluding zeros) ---")
            print(stats)


        print("\n==================== PLOTS ====================")
        for col in text_cols:
            if col not in df.columns:
                continue

            # exclude zero lengths in plot
            lengths = df.loc[df[f"{col}_len"] > 0, f"{col}_len"]

            if lengths.empty:
                continue

            plt.figure()
            lengths.hist(bins=60)
            plt.title(f"Length distribution for {col} (excluding zeros)")
            plt.xlabel("Characters")
            plt.ylabel("Count")
            plt.savefig(f"{col}_length_distribution.pdf")
            plt.close()

        return df


# # -----------------------------
# # MAIN SCRIPT
# # -----------------------------
if __name__ == "__main__":
    parquet_path = "parsed_sec_filings.parquet"

    print("Loading parquet...")
    sec_df = pd.read_parquet(parquet_path)
    
    import pandas as pd

    # Compute lengths
    sec_df["mda_len"] = sec_df["mda_text"].fillna("").astype(str).str.len()
    sec_df["market_risk_len"] = sec_df["market_risk_text"].fillna("").astype(str).str.len()

    # Run analysis
    sec_df_enhanced = analyze_extraction_quality(
        sec_df,
        text_cols=("mda_text", "market_risk_text"),
    )