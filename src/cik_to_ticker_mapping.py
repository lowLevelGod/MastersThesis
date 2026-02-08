import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import json
from sec_form345submission_parser import SecForm345SubmissionParser

class CikTickerMapper:
    def apply_yearly_mapping_to_sec_filings(
        self,
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
        
        self._load_aux_cik_ticker_dict("./edgar_cache/reference/company_tickers.json")
        
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

                resolved = self._resolve_ticker_for_cik(cik, y, yearly_dict, min_year=min_year)
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

    def _load_aux_cik_ticker_dict(self, json_path: str):
        """
        Reads SEC-like JSON file and builds:
        {"0001045810": "NVDA", ...}
        """
        
        if self.aux_cik_map:
            return self.aux_cik_map
        
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

    def _resolve_ticker_for_cik(self, cik: str, year: int, yearly_dict: dict, min_year: int):
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
        
        if cik in self.aux_cik_map:
            return self.aux_cik_map[cik]
        
        return None


if __name__ == "__main__":
    secForm345ubmissionParser = SecForm345SubmissionParser()
    
    secForm345ubmissionParser.parse_form345_folder(
        folder="sec_submissions",
        output_path="cik_ticker_mapping.parquet"
    )

    yearly_dict = secForm345ubmissionParser.build_yearly_cik_ticker_dict("cik_ticker_mapping.parquet")

    cikTickerMapper = CikTickerMapper()
    
    cikTickerMapper.apply_yearly_mapping_to_sec_filings(
        sec_filings_parquet="parsed_sec_filings_cleaned.parquet",
        output_parquet="parsed_sec_filings_with_tickers.parquet",
        yearly_dict=yearly_dict,
        cik_col="cik",
        date_col="filing_date",
        ticker_col="ticker",
        batch_size=50_000
    )