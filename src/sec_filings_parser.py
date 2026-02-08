import re
from pathlib import Path
import os
import pandas as pd
from typing import Optional, Dict
from tqdm import tqdm
import shutil

class SecFilingTxtParser:
    ITEM = r"(?:item|ITEM)\s*"
    SEP = r"(?:\s*[\.\:\-\–\—]?\s*)" 

    # 10-K MD&A (Item 7)
    MDNA_10K_PATTERN = re.compile(
        rf"""
        (
            {ITEM}7{SEP}
            management\s*['’]?\s*s?\s*
            discussion\s*and\s*analysis
            .*?
        )
        (?=
            {ITEM}7a|
            {ITEM}8|
            $ 
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    # 10-K MARKET RISK (Item 7a)
    MARKET_RISK_10K_PATTERN = re.compile(
        rf"""
        (
            {ITEM}7a{SEP}
            quantitative\s*and\s*qualitative\s*
            disclosures\s*about\s*
            market\s*risk
            .*?
        )
        (?=
            {ITEM}8|
            {ITEM}9|
            $ 
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    # 10-Q MD&A (Item 2)
    MDNA_10Q_PATTERN = re.compile(
        rf"""
        (
            {ITEM}2{SEP}
            management\s*['’]?\s*s?\s*
            discussion\s*and\s*analysis
            .*?
        )
        (?=
            {ITEM}3|
            {ITEM}4|
            $ 
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    # 10-Q MARKET RISK (Item 3)
    MARKET_RISK_10Q_PATTERN = re.compile(
        rf"""
        (
            {ITEM}3{SEP}
            quantitative\s*and\s*qualitative\s*
            disclosures\s*about\s*
            market\s*risk
            .*?
        )
        (?=
            {ITEM}4|
            $ 
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )
    
    # ACCEPTANCE-DATETIME is in the header of each filing, e.g.:
    # <ACCEPTANCE-DATETIME>20110103120000</ACCEPTANCE-DATETIME>
    ACCEPTANCE_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")

    def __init__(self, offset: int = 8000, min_section_length: int = 2000):
        self.offset = offset
        self.min_section_length = min_section_length

    def parse_sec_filing(self, filepath: str) -> Optional[Dict]:
        """
        Parse one SEC txt filing file and extract MD&A + Market Risk sections.
        """
        filename = os.path.basename(filepath)

        # Example: 20110103_10-K_edgar_data_779544_0000930413-11-000028.txt
        m = re.match(r"(\d{8})_(10-K|10-Q)_edgar_data_(\d+)_([0-9\-]+)", filename, re.IGNORECASE)
        if not m:
            return None

        filing_date, form_type, cik, accession = m.groups()

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        def extract_acceptance_time(filing_text: str) -> Optional[str]:
            m = self.ACCEPTANCE_RE.search(filing_text[:5000])  # header always near the beginning
            if not m:
                return None

            dt_str = m.group(1)  # YYYYMMDDHHMMSS
            extracted_time = pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", errors="coerce").strftime("%H:%M:%S") # HH:MM:SS
            
            return extracted_time
        
        # Extract acceptance time from the raw text of the filing
        # Used to see whether the filing was accepted during market hours (9:30-16:00) or after hours.
        acceptance_time = extract_acceptance_time(raw_text)
        
        def normalize_text(text: str) -> str:
            text = text.replace("\x00", " ")
            text = re.sub(r"\r\n", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            
            return text.strip()

        raw_text = normalize_text(raw_text)

        if form_type.upper() == "10-K":
            mda_text = self._extract_section_text(
                self.MDNA_10K_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
            market_risk_text = self._extract_section_text(
                self.MARKET_RISK_10K_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
        else:
            mda_text = self._extract_section_text(
                self.MDNA_10Q_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
            market_risk_text = self._extract_section_text(
                self.MARKET_RISK_10Q_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )

        return {
            "filing_date": pd.to_datetime(filing_date),
            "acceptance_time": acceptance_time,
            "form_type": form_type.upper(),
            "cik": cik,
            "accession": accession,
            "mda_text": mda_text if mda_text else "",
            "market_risk_text": market_risk_text if market_risk_text else "",
        }

    def parse_folder(self, folder: str, output_parquet: str) -> pd.DataFrame:
        """
        Recursively parse all txt files in folder.
        Writes each parsed result immediately as a small parquet chunk.
        At the end, merges all chunks into one parquet file.
        """
        folder = str(folder)
        output_parquet = str(output_parquet)

        filings_paths = list(Path(folder).rglob("*.txt"))

        # temp chunk folder
        chunk_dir = Path(output_parquet).with_suffix("").as_posix() + "_chunks"
        chunk_dir = Path(chunk_dir)

        # clean old output
        if os.path.exists(output_parquet):
            os.remove(output_parquet)

        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_count = 0

        mda_not_found = 0
        market_risk_not_found = 0
        
        for path in tqdm(filings_paths, total=len(filings_paths), desc="Parsing SEC filings"):
            parsed = self.parse_file(path)

            if parsed:
                df_row = pd.DataFrame([parsed])

                chunk_path = chunk_dir / f"chunk_{chunk_count:07d}.parquet"
                df_row.to_parquet(chunk_path, index=False)

                chunk_count += 1
                
                if not parsed["mda_text"]:
                    mda_not_found += 1
                    
                if not parsed["market_risk_text"]:
                    market_risk_not_found += 1
            
        print(f"Total filings parsed: {chunk_count}")
        print(f"MD&A section not found in {mda_not_found} filings ({mda_not_found / chunk_count:.2%})")
        print(f"Market Risk section not found in {market_risk_not_found} filings ({market_risk_not_found / chunk_count:.2%})")

        if chunk_count == 0:
            return pd.DataFrame()

        # Merge all parquet chunks
        all_chunks = list(chunk_dir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in all_chunks], ignore_index=True)

        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.sort_values("filing_date").reset_index(drop=True)

        # Save final parquet
        df.to_parquet(output_parquet, index=False)

        # cleanup
        shutil.rmtree(chunk_dir)

        return df
    
    def _extract_section_text(
        self,
        pattern: re.Pattern,
        text: str,
        offset: int = 8000,
        min_len: int = 2000
    ) -> Optional[str]:
        def remove_heading(section_text: str) -> str:
            """
            Remove the initial 'Item X ...' heading line if present.
            """
            section_text = section_text.strip()

            HEADING_CLEAN_PATTERN = re.compile(
                r"^\s*item\s*\d+[a-z]?\s*[\.\:\-\–\—]?\s*.*?\n+",
                re.IGNORECASE
            )
            # remove first "Item X ..." line
            section_text = re.sub(HEADING_CLEAN_PATTERN, "", section_text)

            # sometimes heading is inline without newline -> remove first ~200 chars if it starts with Item
            section_text = re.sub(
                r"^\s*item\s*\d+[a-z]?\s*[\.\:\-\–\—]?\s*",
                "",
                section_text,
                flags=re.IGNORECASE
            )

            return section_text.strip()

        def remove_noisy_lines(text: str, min_chars: int = 50) -> str:
            """
            Remove short lines from extracted section text and lines that are mostly numeric or separators (common in tables).
            """

            lines = text.splitlines()
            cleaned_lines = []

            for line in lines:
                stripped = line.strip()

                # drop very short lines
                if not stripped or len(stripped) < min_chars:
                    continue
                
                has_letter = any(c.isalpha() for c in stripped)

                # drop numeric/separator lines
                if not has_letter:
                    continue

                cleaned_lines.append(stripped)
                    
            if len(cleaned_lines) < 2:
                return ""
            
            return "\n".join(cleaned_lines).strip()
        
        # Try search after offset (avoid TOC)
        if len(text) > offset:
            match = pattern.search(text[offset:])

        if match:
            extracted = match.group(0).strip()
        else:
            # Fallback: take longest match in full text
            matches = list(pattern.finditer(text))
            if not matches:
                return None

            best = max(matches, key=lambda m: len(m.group(0)))
            extracted = best.group(0).strip()

        extracted = remove_heading(extracted)
        extracted = remove_noisy_lines(extracted)

        if len(extracted) < min_len:
            return None

        return extracted


if __name__ == "__main__":
    parser = SecFilingTxtParser(offset=8000, min_section_length=2000)

    sec_df = parser.parse_folder("./sec_filings", "parsed_sec_filings.parquet")

