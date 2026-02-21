import pandas as pd

INPUT_FILE = "romanian_stock_prices.txt"      # change if needed
OUTPUT_FILE = "romanian_tickers.txt"

# Read the file as a tab-separated dataset
df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str)

# Extract distinct symbols (drop NaNs just in case)
symbols = sorted(df["Symbol"].dropna().unique())

# Save to output file, one symbol per line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sym in symbols:
        f.write(sym.strip() + "\n")

print(f"Extracted {len(symbols)} unique symbols.")
print(f"Saved to: {OUTPUT_FILE}")