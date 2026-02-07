import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import re
from tqdm import tqdm


ACCEPTANCE_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")

def extract_acceptance_datetime_from_file(filepath: Path):
    """
    Reads only the first part of the file and extracts ACCEPTANCE-DATETIME.
    Returns pandas.Timestamp or None.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(5000)  # header always near the beginning

        m = ACCEPTANCE_RE.search(head)
        if not m:
            return None

        dt_str = m.group(1)  # YYYYMMDDHHMMSS
        return pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", errors="coerce")

    except Exception:
        return None


def build_expected_filename(filing_date, form_type, cik, accession_number):
    """
    Example format:
    20180402_10-K_edgar_data_313364_0001047469-18-002446.txt
    """
    date_str = pd.to_datetime(filing_date).strftime("%Y%m%d")
    return f"{date_str}_{form_type}_edgar_data_{cik}_{accession_number}.txt"


def locate_file(root_folder: str, filing_date, filename: str):
    """
    Folder structure is root/year/QTRq/...
    We search only within root/year/ to avoid expensive global recursion.
    """
    filing_year = pd.to_datetime(filing_date).year
    year_folder = Path(root_folder) / str(filing_year)

    if not year_folder.exists():
        return None

    # Search inside all QTR folders
    for qtr_folder in year_folder.glob("QTR*"):
        candidate = qtr_folder / filename
        if candidate.exists():
            return candidate

    return None


def add_acceptance_times(
    input_parquet: str,
    filings_root: str,
    output_parquet: str,
    batch_size: int = 50000
):
    """
    Reads parquet in batches, locates raw txt filing file, extracts ACCEPTANCE-DATETIME,
    writes enhanced parquet.
    """
    pf = pq.ParquetFile(input_parquet)

    writer = None
    total_rows = pf.metadata.num_rows

    print(f"Total rows in parquet: {total_rows:,}")

    with tqdm(total=total_rows, desc="Processing filings") as pbar:
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            acceptance_datetimes = []

            for _, row in df.iterrows():
                filing_date = row["filing_date"]
                form_type = row["form_type"]
                # CIKs in filenames have no leading zeros, but in parquet they do, so we strip them here.
                cik = row["cik"].lstrip("0")
                accession_number = row["accession"]

                filename = build_expected_filename(
                    filing_date=filing_date,
                    form_type=form_type,
                    cik=cik,
                    accession_number=accession_number
                )

                filepath = locate_file(
                    root_folder=filings_root,
                    filing_date=filing_date,
                    filename=filename
                )

                if filepath is None:
                    acceptance_datetimes.append(pd.NaT)
                    continue

                dt = extract_acceptance_datetime_from_file(filepath)
                acceptance_datetimes.append(dt if dt is not None else pd.NaT)

            df["acceptance_datetime"] = acceptance_datetimes
            df["acceptance_time"] = df["acceptance_datetime"].dt.strftime("%H:%M:%S")
            df.drop(columns=["acceptance_datetime"], inplace=True)

            # drop rows that have ticker None or empty, as they are not useful for our analysis
            df = df[df["ticker"].notna()].copy()
            # remove NASDAQ:, NYSE: etc prefixes if they exist, as they are not useful for our analysis and can cause issues with matching tickers to price data
            df["ticker"] = df["ticker"].str.replace(r"^[A-Za-z]+:", "", regex=True)
            # strip whtespace 
            df["ticker"] = df["ticker"].str.strip()
            # drop rows that have tickers that contain characters that are not valid for stock tickers, as they are likely to be errors
            df = df[~df["ticker"].str.contains(r"[^A-Za-z0-9\.\-]", regex=True)].copy()
            
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)

            writer.write_table(table)

            pbar.update(len(df))

    if writer is not None:
        writer.close()

    print(f"\nSaved enhanced parquet to: {output_parquet}")


if __name__ == "__main__":
    INPUT_PARQUET = "parsed_sec_filings_with_tickers.parquet"
    FILINGS_ROOT = "sec_filings/"  # root/year/QTRx/*.txt
    OUTPUT_PARQUET = "parsed_sec_filings_with_tickers_and_time.parquet"

    add_acceptance_times(
        input_parquet=INPUT_PARQUET,
        filings_root=FILINGS_ROOT,
        output_parquet=OUTPUT_PARQUET,
        batch_size=20000
    )