from pathlib import Path
from tqdm import tqdm
import pandas as pd 

class SecForm345SubmissionParser:  
    def parse_form345_folder(
        self,
        folder: str,
        output_path: str
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
        combined.to_parquet(output_path, index=False)
        print(f"\nSaved {len(combined):,} records to {output_path}")
        
        return combined


    def build_yearly_cik_ticker_dict(self, mapping_parquet_path: str):
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