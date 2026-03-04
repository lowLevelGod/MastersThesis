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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FuncFormatter

def set_publication_style():
    sns.set_theme(style="white")  # clean white background
    
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
    })

def thousands_formatter(x, pos):
    return f"{int(x):,}"

def analyze_extraction_quality(
    sec_df: pd.DataFrame,
    text_cols=("mda_text", "market_risk_text"),
):
    """
    Exploratory Data Analysis for SEC text extraction quality.
    Produces descriptive statistics and publication-ready plots.
    """

    set_publication_style()
    
    df = sec_df.copy()

    print("\n==================== BASIC DATASET INFO ====================")
    print(f"Number of rows: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    print("Columns:")
    print(df.columns.tolist())

    # ----------------------------------------------------------
    # Ensure datetime
    # ----------------------------------------------------------
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    # ----------------------------------------------------------
    # 1️⃣ Missing Rates (empty string = missing)
    # ----------------------------------------------------------
    print("\n==================== MISSING RATE (%) ====================")

    for col in text_cols:
        if col not in df.columns:
            print(f"{col}: column not found")
            continue

        missing_mask = df[col].fillna("").astype(str).str.strip() == ""
        missing_rate = missing_mask.mean() * 100

        print(f"{col}: {missing_rate:.2f}% missing")

    # ----------------------------------------------------------
    # 2️⃣ Character Length Statistics (Paper-ready formatting)
    # ----------------------------------------------------------
    print("\n==================== LENGTH STATISTICS ====================")

    for col in text_cols:
        if col not in df.columns:
            continue

        df[f"{col}_len"] = df[col].fillna("").astype(str).str.len()
        
        lengths = df[f"{col}_len"]
        
        positive_lengths = lengths[lengths > 0]

        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        stats = positive_lengths.describe(percentiles=percentiles)

        print(f"\n--- {col} character length ---")
        print(f"Mean: {stats['mean']:,.0f}")
        print(f"Std:  {stats['std']:,.0f}")
        print(f"Min:  {stats['min']:,.0f}")
        print(f"10th percentile: {stats['10%']:,.0f}")
        print(f"25th percentile: {stats['25%']:,.0f}")
        print(f"Median: {stats['50%']:,.0f}")
        print(f"75th percentile: {stats['75%']:,.0f}")
        print(f"90th percentile: {stats['90%']:,.0f}")
        print(f"Max:  {stats['max']:,.0f}")

    # ----------------------------------------------------------
    # 3️⃣ Character Length Distributions
    # ----------------------------------------------------------
    print("\nGenerating length distribution plots...")

    for col in text_cols:
        if col not in df.columns:
            continue

        lengths = df[f"{col}_len"]
        
        positive_lengths = lengths[lengths > 0]

        plt.figure()
        sns.histplot(positive_lengths, bins=60)

        plt.xscale("log")

        plt.title(f"Log-Scale Distribution of {col.replace('_', ' ').upper()} Length")
        plt.xlabel("Number of Characters (log scale)")
        plt.ylabel("Frequency")
        
        median_val = positive_lengths.median()
        mean_val = positive_lengths.mean()

        plt.axvline(median_val, linestyle="--", linewidth=1, label="Median")
        plt.axvline(mean_val, linestyle=":", linewidth=1, label="Mean")
        plt.legend(frameon=False)

        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        plt.tight_layout()
        plt.savefig(f"{col}_length_distribution_log.pdf", bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    # 4️⃣ Filings Per Year Distribution
    # ----------------------------------------------------------
    if "filing_date" in df.columns:
        df["year"] = df["filing_date"].dt.year
        filings_per_year = df["year"].value_counts().sort_index()

        plt.figure()
        filings_per_year.plot(kind="bar", width=0.8)

        plt.title("Number of SEC Filings per Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Filings")

        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        plt.tight_layout()
        plt.savefig("filings_per_year_distribution.pdf", bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    # 5️⃣ Missing Rate by Form Type (10-K vs 10-Q)
    # ----------------------------------------------------------
    if "form_type" in df.columns:
        missing_data = []

        for form in df["form_type"].unique():
            df_form = df[df["form_type"] == form]

            for col in text_cols:
                if col not in df_form.columns:
                    continue

                missing_mask = df_form[col].fillna("").astype(str).str.strip() == ""
                missing_rate = missing_mask.mean() * 100

                missing_data.append({
                    "form_type": form,
                    "section": col.replace("_text", "").upper(),
                    "missing_rate": missing_rate
                })

        missing_df = pd.DataFrame(missing_data)

        plt.figure()
        sns.barplot(
            data=missing_df,
            x="form_type",
            y="missing_rate",
            hue="section"
        )

        plt.title("Missing Section Rate by Form Type")
        plt.xlabel("Form Type")
        plt.ylabel("Missing Rate (%)")

        plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")

        plt.legend(title="Section", frameon=False)

        plt.tight_layout()
        plt.savefig("missing_rate_by_form_type.pdf", bbox_inches="tight")
        plt.close()

    print("\nEDA complete.")
    
    return df

if __name__ == "__main__":
    parquet_path = "parsed_sec_filings_cleaned.parquet"

    print("Loading parquet...")
    sec_df = pd.read_parquet(parquet_path)

    sec_df_enhanced = analyze_extraction_quality(sec_df)