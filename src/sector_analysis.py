import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_sector_industry_structure(
    df: pd.DataFrame,
    ticker_col="ticker",
    sector_col="sector",
    industry_col="industry",
):
    """
    Performs sector and industry-level exploratory analysis.
    Includes firm-level and filing-level distributions.
    Produces publication-quality plots.
    """

    set_publication_style()
    df = df.copy()

    print("\n==================== SECTOR & INDUSTRY ANALYSIS ====================")

    # ----------------------------------------------------------
    # Basic Counts
    # ----------------------------------------------------------
    n_sectors = df[sector_col].nunique(dropna=True)
    n_industries = df[industry_col].nunique(dropna=True)

    print(f"Unique sectors: {n_sectors}")
    print(f"Unique industries (non-missing): {n_industries}")

    # ----------------------------------------------------------
    # Sector Distribution (Firm-Level)
    # ----------------------------------------------------------
    print("\n==================== SECTOR DISTRIBUTION (FIRM LEVEL) ====================")

    firm_sector = (
        df[[ticker_col, sector_col]]
        .drop_duplicates()
        .groupby(sector_col)[ticker_col]
        .count()
        .sort_values(ascending=False)
    )

    print(firm_sector)

    # Bar plot — clean and readable
    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=firm_sector.values,
        y=firm_sector.index,
        orient="h"
    )

    plt.title("Number of Companies per Sector")
    plt.xlabel("Number of Companies")
    plt.ylabel("Sector")

    plt.tight_layout()
    plt.savefig("companies_per_sector.pdf", bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # Sector Distribution (Filing-Level)
    # ----------------------------------------------------------
    print("\n==================== SECTOR DISTRIBUTION (FILING LEVEL) ====================")

    filing_sector = (
        df[sector_col]
        .value_counts()
        .sort_values(ascending=False)
    )

    print(filing_sector)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=filing_sector.values,
        y=filing_sector.index,
        orient="h"
    )

    plt.title("Number of Filings per Sector")
    plt.xlabel("Number of Filings")
    plt.ylabel("Sector")

    plt.tight_layout()
    plt.savefig("filings_per_sector.pdf", bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # Industry Distribution (Firm-Level, Top 20 Only)
    # ----------------------------------------------------------
    print("\n==================== INDUSTRY DISTRIBUTION (FIRM LEVEL) ====================")

    firm_industry = (
        df[[ticker_col, industry_col]]
        .dropna()
        .drop_duplicates()
        .groupby(industry_col)[ticker_col]
        .count()
        .sort_values(ascending=False)
    )

    print("Top 20 industries by number of companies:")
    print(firm_industry.head(20))

    # Plot only Top 20 industries for readability
    top20_industries = firm_industry.head(20)

    plt.figure(figsize=(9, 7))
    sns.barplot(
        x=top20_industries.values,
        y=top20_industries.index,
        orient="h"
    )

    plt.title("Top 20 Industries by Number of Companies")
    plt.xlabel("Number of Companies")
    plt.ylabel("Industry")

    plt.tight_layout()
    plt.savefig("top20_industries_by_companies.pdf", bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # Long-Tail Industry Structure
    # ----------------------------------------------------------
    print("\n==================== INDUSTRY LONG-TAIL STRUCTURE ====================")

    industry_counts = firm_industry

    print(f"Median companies per industry: {industry_counts.median():.2f}")
    print(f"10th percentile: {industry_counts.quantile(0.1):.2f}")
    print(f"90th percentile: {industry_counts.quantile(0.9):.2f}")
    print(f"Maximum: {industry_counts.max()}")

    # Log-scale histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(industry_counts, bins=40)
    plt.xscale("log")

    plt.title("Distribution of Companies per Industry (Log Scale)")
    plt.xlabel("Number of Companies (log scale)")
    plt.ylabel("Number of Industries")

    plt.tight_layout()
    plt.savefig("industry_company_distribution_log.pdf", bbox_inches="tight")
    plt.close()
    
    print("\nSector & industry analysis complete.")

    return firm_sector, firm_industry

if __name__ == "__main__":
    parquet_path = "us_market_dataset_with_sector.parquet"

    print("Loading parquet...")
    sec_df = pd.read_parquet(parquet_path)

    sec_df_enhanced = analyze_sector_industry_structure(sec_df)