import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_form345_folder(
    folder: str,
    output_path: str = "form345_companies.parquet"
):
    folder = Path(folder)

    files = sorted(folder.glob("*_submission.tsv"))

    if not files:
        raise FileNotFoundError(f"No *_submission.tsv files found in {folder}")

    rows = []

    for file in tqdm(files, desc="Parsing submission files"):
        try:
            df = pd.read_csv(file, sep="\t", dtype=str, low_memory=False)

            # Keep only needed columns
            keep_cols = ["FILING_DATE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL"]

            missing = [c for c in keep_cols if c not in df.columns]
            if missing:
                print(f"[WARN] Skipping {file.name}, missing columns: {missing}")
                continue

            df = df[keep_cols].copy()

            df = df.rename(columns={
                "FILING_DATE": "filing_date",
                "ISSUERCIK": "cik",
                "ISSUERNAME": "company_name",
                "ISSUERTRADINGSYMBOL": "ticker"
            })

            df["filing_date"] = pd.to_datetime(df["filing_date"],  format="%d-%b-%Y", errors="coerce")

            # Normalize cik formatting
            df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)

            # Normalize tickers
            df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

            # Remove empty tickers
            df.loc[df["ticker"].isin(["", "NONE", "NAN"]), "ticker"] = None

            rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed parsing {file.name}: {e}")

    if not rows:
        raise RuntimeError("No data was extracted from the folder.")

    combined = pd.concat(rows, ignore_index=True)

    # Optional: remove invalid rows
    combined = combined.dropna(subset=["cik", "filing_date"])

    # Optional: drop duplicates
    combined = combined.drop_duplicates(subset=["cik", "ticker", "company_name", "filing_date"])

    combined = combined.sort_values("filing_date").reset_index(drop=True)

    # Save
    output_path = Path(output_path)

    if output_path.suffix == ".csv":
        combined.to_csv(output_path, index=False)
    else:
        combined.to_parquet(output_path, index=False)

    print(f"\nSaved {len(combined):,} records to {output_path}")
    return combined

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def build_yearly_cik_ticker_dict(mapping_parquet_path: str):
    """
    Builds:
        { year: { cik: ticker } }
    from cik_ticker_mapping.parquet
    """
    df = pd.read_parquet(mapping_parquet_path)

    required_cols = {"filing_date", "cik", "ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in mapping file: {missing}")

    df = df.dropna(subset=["filing_date", "cik", "ticker"]).copy()

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["filing_date"])

    df["year"] = df["filing_date"].dt.year
    df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Sort so that the latest filing_date is last (so dict overwrite keeps most recent)
    df = df.sort_values(["year", "cik", "filing_date"])

    yearly_dict = {}

    for year, group in df.groupby("year"):
        cik_map = {}
        for _, row in group.iterrows():
            cik_map[row["cik"]] = row["ticker"]
        yearly_dict[int(year)] = cik_map

    print(f"Built yearly dictionary for {len(yearly_dict)} years.")
    return yearly_dict

import json

def load_aux_cik_ticker_dict(json_path: str):
    """
    Reads SEC-like JSON file and builds:
      {"0001045810": "NVDA", ...}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    aux = {}

    for _, record in data.items():
        cik = str(record.get("cik_str", "")).strip()
        ticker = str(record.get("ticker", "")).strip().upper()

        if not cik or not ticker:
            continue

        cik = cik.zfill(10)
        aux[cik] = ticker

    print(f"Loaded auxiliary CIK->ticker mapping: {len(aux):,} entries.")
    
    return aux

aux_cik_map = load_aux_cik_ticker_dict("./edgar_cache/reference/company_tickers.json")

def resolve_ticker_for_cik(cik: str, year: int, yearly_dict: dict, min_year: int):
    """
    Search year downwards until match found.
    """
    for y in range(year, min_year - 1, -1):
        year_map = yearly_dict.get(y)
        if year_map is None:
            continue
        t = year_map.get(cik)
        if t:
            return t
    
    if cik in aux_cik_map:
        return aux_cik_map[cik]
    
    return None


def apply_yearly_mapping_to_sec_filings(
    sec_filings_parquet: str,
    output_parquet: str,
    yearly_dict: dict,
    cik_col: str = "cik",
    date_col: str = "filing_date",
    ticker_col: str = "ticker",
    batch_size: int = 50_000
):
    """
    Iterates SEC filings parquet in batches and assigns ticker using yearly mapping.
    """
    pf = pq.ParquetFile(sec_filings_parquet)

    total_rows = pf.metadata.num_rows
    min_year = min(yearly_dict.keys())
    max_year = max(yearly_dict.keys())

    print(f"SEC filings total rows: {total_rows:,}")
    print(f"Mapping year range: {min_year} -> {max_year}")

    writer = None

    # Cache results for speed:
    # (cik, year) -> ticker
    cache = {}

    pbar = tqdm(total=total_rows, desc="Mapping CIK -> ticker")

    total_filings = 0
    ciks_resolved = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        
        total_filings += len(df)

        if cik_col not in df.columns:
            raise ValueError(f"Missing '{cik_col}' column in SEC filings parquet")
        if date_col not in df.columns:
            raise ValueError(f"Missing '{date_col}' column in SEC filings parquet")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[cik_col] = df[cik_col].astype(str).str.strip().str.zfill(10)

        years = df[date_col].dt.year

        tickers = []
        for cik, y in zip(df[cik_col], years):
            if pd.isna(y):
                tickers.append(None)
                continue

            y = int(y)

            key = (cik, y)
            if key in cache:
                tickers.append(cache[key])
                continue

            resolved = resolve_ticker_for_cik(cik, y, yearly_dict, min_year=min_year)
            cache[key] = resolved
            tickers.append(resolved)

        ciks_resolved += sum(1 for t in tickers if t is not None)
        df[ticker_col] = tickers

        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(output_parquet, table.schema)

        writer.write_table(table)

        pbar.update(len(df))

    pbar.close()

    if writer:
        writer.close()

    print(f"Total filings processed: {total_filings:,}")
    print(f"Total CIKs resolved to tickers: {ciks_resolved:,}")
    print(f"Saved updated parquet: {output_parquet}")
    print(f"Cache size: {len(cache):,}")


if __name__ == "__main__":
    # -------------------------------------------------------
    # Step 1: Build yearly dict from cik_ticker_mapping.parquet
    # -------------------------------------------------------
    yearly_dict = build_yearly_cik_ticker_dict("cik_ticker_mapping.parquet")

    # -------------------------------------------------------
    # Step 2: Apply mapping to SEC filings parquet
    # -------------------------------------------------------
    apply_yearly_mapping_to_sec_filings(
        sec_filings_parquet="parsed_sec_filings_cleaned.parquet",
        output_parquet="parsed_sec_filings_with_tickers.parquet",
        yearly_dict=yearly_dict,
        cik_col="cik",
        date_col="filing_date",
        ticker_col="ticker",
        batch_size=50_000
    )

# if __name__ == "__main__":
    # df = parse_form345_folder(
    #     folder="sec_submissions",
    #     output_path="cik_ticker_mapping.parquet"
    # )

    # print(df.head())