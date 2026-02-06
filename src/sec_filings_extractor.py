import re
from pathlib import Path
import os
import pandas as pd
from typing import Optional, Dict
from typing import Dict, Optional, List
from tqdm import tqdm
import shutil

# -----------------------------
# Regex patterns
# -----------------------------

ITEM = r"(?:item|ITEM)\s*"
SEP = r"(?:\s*[\.\:\-\–\—]?\s*)"  # dot/colon/dash separators

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


# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


import re
from typing import Optional

HEADING_CLEAN_PATTERN = re.compile(
    r"^\s*item\s*\d+[a-z]?\s*[\.\:\-\–\—]?\s*.*?\n+",
    re.IGNORECASE
)

def remove_heading(section_text: str) -> str:
    """
    Remove the initial 'Item X ...' heading line if present.
    """
    section_text = section_text.strip()

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

def remove_non_letter_lines(text: str, max_len: int = 40) -> str:
    """
    Remove short lines that contain no alphabetic characters.
    Keeps longer lines even if numeric-heavy.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        has_letter = any(c.isalpha() for c in stripped)

        # drop short numeric/separator lines
        if not has_letter and len(stripped) <= max_len:
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned).strip()

def remove_short_lines(text: str, min_chars: int = 50) -> str:
    """
    Remove short lines from extracted section text.
    Keeps lines that contain important financial keywords even if short.
    """

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Always keep longer lines
        if len(stripped) >= min_chars:
            cleaned.append(stripped)
            continue

    if len(cleaned) < 2:
        return ""
    
    return "\n".join(cleaned).strip()

def extract_section(
    pattern: re.Pattern,
    text: str,
    offset: int = 8000,
    min_len: int = 2000
) -> Optional[str]:

    text = normalize_text(text)

    # 1) Try search after offset (avoid TOC)
    if len(text) > offset:
        match = pattern.search(text[offset:])
        if match:
            extracted = match.group(0).strip()
            extracted = remove_heading(extracted)
            extracted = remove_non_letter_lines(extracted)

            if len(extracted) >= min_len:
                return extracted

    # 2) Fallback: take longest match in full text
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    best = max(matches, key=lambda m: len(m.group(0)))
    extracted = best.group(0).strip()

    extracted = remove_heading(extracted)
    extracted = remove_non_letter_lines(extracted)
    extracted = remove_short_lines(extracted)

    if len(extracted) < min_len:
        return None

    return extracted

# -----------------------------
# Filing parser class
# -----------------------------
class SecFilingTxtParser:
    def __init__(self, offset: int = 8000, min_section_length: int = 2000):
        self.offset = offset
        self.min_section_length = min_section_length

    def parse_file(self, filepath: str) -> Optional[Dict]:
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

        raw_text = normalize_text(raw_text)

        if form_type.upper() == "10-K":
            mda_text = extract_section(
                MDNA_10K_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
            market_risk_text = extract_section(
                MARKET_RISK_10K_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
        else:
            mda_text = extract_section(
                MDNA_10Q_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )
            market_risk_text = extract_section(
                MARKET_RISK_10Q_PATTERN,
                raw_text,
                offset=self.offset,
                min_len=self.min_section_length
            )

        # print(f"Parsing file: {filename}")
        # if mda_text:
        #     print(f"Extracted MDA text: {mda_text.splitlines()}")
        # else:
        #     print("MDA section not found.")
        
        # if market_risk_text:
        #     print(f"Extracted Market Risk text: {market_risk_text}")
        # else:
        #     print("Market Risk section not found.")

        return {
            "filing_date": pd.to_datetime(filing_date),
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


# ============================================================
# 6. Example usage
# ============================================================

# if __name__ == "__main__":
#     parser = SecFilingTxtParser(offset=8000, min_section_length=2000)

#     sec_df = parser.parse_folder("./sec_filings", "parsed_sec_filings.parquet")

#     print(sec_df.head())
#     print("\nMissing rates:")
#     print(sec_df[["has_mda", "has_market_risk"]].mean())
    
# Total filings parsed: 398522
# MD&A section not found in 42764 filings (10.73%)
# Market Risk section not found in 298560 filings (74.92%)

sec_df = pd.read_parquet("parsed_sec_filings.parquet")

