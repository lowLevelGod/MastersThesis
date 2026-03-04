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

def analyze_company_structure(
    df: pd.DataFrame,
    ticker_col="ticker",
    date_col="filing_date",
):
    """
    Company-level exploratory analysis after CIK → ticker mapping.
    Produces firm-level descriptive statistics and publication-quality plots.
    """
    
    set_publication_style()
    
    df = df.copy()

    print("\n==================== COMPANY-LEVEL ANALYSIS ====================")

    # ----------------------------------------------------------
    # Basic Stats
    # ----------------------------------------------------------
    n_rows = len(df)
    n_companies = df[ticker_col].nunique()

    print(f"Total filings: {n_rows:,}")
    print(f"Unique companies (tickers): {n_companies:,}")
    print(f"Average filings per company: {n_rows / n_companies:,.2f}")

    # ----------------------------------------------------------
    # Filings per Company
    # ----------------------------------------------------------
    filings_per_company = df[ticker_col].value_counts()

    print("\n==================== FILINGS PER COMPANY ====================")
    print(f"Median filings per company: {filings_per_company.median():,.0f}")
    print(f"10th percentile: {filings_per_company.quantile(0.1):,.0f}")
    print(f"25th percentile: {filings_per_company.quantile(0.25):,.0f}")
    print(f"75th percentile: {filings_per_company.quantile(0.75):,.0f}")
    print(f"90th percentile: {filings_per_company.quantile(0.9):,.0f}")
    print(f"Maximum filings for a single company: {filings_per_company.max():,}")

    # ----------------------------------------------------------
    # Log-scale Distribution Plot (much cleaner)
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(filings_per_company, bins=40)
    plt.xscale("log")

    plt.title("Distribution of Filings per Company (Log Scale)")
    plt.xlabel("Number of Filings (log scale)")
    plt.ylabel("Number of Companies")

    plt.tight_layout()
    plt.savefig("filings_per_company_distribution_log.pdf", bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # Concentration Analysis (Top Companies)
    # ----------------------------------------------------------
    print("\n==================== CONCENTRATION ====================")

    top_1_share = filings_per_company.iloc[0] / n_rows
    top_5_share = filings_per_company.iloc[:5].sum() / n_rows
    top_10_share = filings_per_company.iloc[:10].sum() / n_rows

    print(f"Top 1 company share of filings: {top_1_share:.2%}")
    print(f"Top 5 companies share: {top_5_share:.2%}")
    print(f"Top 10 companies share: {top_10_share:.2%}")

    # ----------------------------------------------------------
    # Time Coverage per Company
    # ----------------------------------------------------------
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        coverage = (
            df.groupby(ticker_col)[date_col]
            .agg(["min", "max"])
            .dropna()
        )

        coverage["years_covered"] = (
            (coverage["max"] - coverage["min"]).dt.days / 365.25
        )

        print("\n==================== TIME COVERAGE ====================")
        print(f"Median years covered per company: {coverage['years_covered'].median():.2f}")
        print(f"90th percentile: {coverage['years_covered'].quantile(0.9):.2f}")
        print(f"Maximum coverage: {coverage['years_covered'].max():.2f}")

        # Plot coverage distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(coverage["years_covered"], bins=40)

        plt.title("Distribution of Company Time Coverage")
        plt.xlabel("Years Covered")
        plt.ylabel("Number of Companies")

        plt.tight_layout()
        plt.savefig("company_time_coverage_distribution.pdf", bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------
    # Panel Balance Indicator
    # ----------------------------------------------------------
    print("\n==================== PANEL BALANCE ====================")

    filings_std = filings_per_company.std()
    filings_mean = filings_per_company.mean()
    coeff_var = filings_std / filings_mean

    print(f"Std of filings per company: {filings_std:,.2f}")
    print(f"Coefficient of variation: {coeff_var:.2f}")

    print("\nCompany-level EDA complete.")

    return filings_per_company

if __name__ == "__main__":
    parquet_path = "us_market_dataset_finbert.parquet"

    print("Loading parquet...")
    sec_df = pd.read_parquet(parquet_path)

    sec_df_enhanced = analyze_company_structure(sec_df)