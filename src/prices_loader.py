import pandas as pd
from pathlib import Path
from tqdm import tqdm

SEC_FILINGS_PARQUET = "parsed_sec_filings_with_tickers_and_time.parquet"
STOOQ_FOLDER = "stooq_prices"   # <-- folder containing cdr_b.us.txt etc
OUTPUT_PARQUET = "stooq_prices_2010_2024.parquet"

# take from 2010 to have some history before our SEC filings data starts, to be able to compute features like 200-day moving average at the start of our SEC filings data in 2011.
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def normalize_stooq_filename_ticker(ticker: str) -> str:
    """
    SEC ticker format -> stooq filename format:
    - lowercase
    - replace "-" with "_"
    """
    t = str(ticker).strip().lower()
    t = t.replace("-", "_")
    return t


def load_stooq_file(path: Path) -> pd.DataFrame:
    """
    Load one stooq txt file.
    Format:
    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    """
    df = pd.read_csv(path)

    # normalize columns
    df.columns = [c.strip("<>").strip().lower() for c in df.columns]

    # parse date
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # keep daily only (PER == D)
    df = df[df["per"] == "D"].copy()

    # rename + keep only what we need
    df = df.rename(columns={
        "ticker": "ticker_raw",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "volume",
    })

    df = df[["ticker_raw", "date", "open", "high", "low", "close", "volume"]]

    return df


def main():
    print("Loading SEC filings parquet...")
    sec_df = pd.read_parquet(SEC_FILINGS_PARQUET)

    tickers = (
        sec_df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"None": None, "nan": None, "": None})
        .dropna()
        .unique()
        .tolist()
    )

    tickers = sorted(set(tickers))
    print(f"Unique tickers found in SEC filings: {len(tickers):,}")

    stooq_folder = Path(STOOQ_FOLDER)
    if not stooq_folder.exists():
        raise FileNotFoundError(f"Folder not found: {stooq_folder}")

    all_prices = []
    missing_files = []

    for ticker in tqdm(tickers, desc="Loading Stooq files"):
        stooq_name = normalize_stooq_filename_ticker(ticker)
        file_path = stooq_folder / f"{stooq_name}.us.txt"

        if not file_path.exists():
            missing_files.append(ticker)
            continue

        try:
            df = load_stooq_file(file_path)

            if df.empty:
                continue

            # add SEC-style ticker
            df["ticker"] = ticker

            # filter date range
            df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

            if df.empty:
                continue

            all_prices.append(df)

        except Exception as e:
            print(f"Failed parsing {file_path.name}: {e}")
            continue

    if not all_prices:
        print("No price data found.")
        return

    prices_df = pd.concat(all_prices, ignore_index=True)
    prices_df.drop(columns=["ticker_raw"], inplace=True) 
    
    # sort
    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(prices_df.head())

    print(f"\nFinal rows: {len(prices_df):,}")
    print(f"Tickers with price data: {prices_df['ticker'].nunique():,}")
    print(f"Tickers missing files: {len(missing_files):,}")

    prices_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved parquet: {OUTPUT_PARQUET}")

    # optional: save missing tickers list
    if missing_files:
        pd.Series(missing_files, name="missing_ticker").to_csv("missing_stooq_tickers.csv", index=False)
        print("Saved missing tickers list to missing_stooq_tickers.csv")


if __name__ == "__main__":
    main()