import os 
from sec_filings_parser import SecFilingTxtParser
from sec_filings_outlier_removal import SecFilingsOutlierRemover
from cik_to_ticker_mapping import CikTickerMapper
from sec_form345submission_parser import SecForm345SubmissionParser
from prices_loader import StockPricesLoader
from price_sec_filings_combiner import PriceSecFilingsCombiner
from finbert_sentiment_enhancer import FinBertSentimentEnhancer

# 1. Parse the SEC filings if not already parsed.
if not os.path.exists("parsed_sec_filings.parquet"):
   parser = SecFilingTxtParser(offset=8000, min_section_length=2000)
   parser.parse_folder("./sec_filings", "parsed_sec_filings.parquet")

# 2. Clean the parsed data by removing outliers and rows where both MDA and Market Risk are missing.
if not os.path.exists("parsed_sec_filings_cleaned.parquet"):
    secFilingsOutlierRemover = SecFilingsOutlierRemover()
    
    secFilingsOutlierRemover.remove_outliers(
        input_parquet = "parsed_sec_filings.parquet",
        output_parquet = "parsed_sec_filings_cleaned.parquet",
        percentile_cutoff = 0.99
    )
    
# 3. Enhanced parsed SEC filings with associated tickers from CIKs using the 3/4/5 forms submission data from SEC.
if not os.path.exists("parsed_sec_filings_with_tickers.parquet"):
    
    if not os.path.exists("cik_ticker_mapping.parquet"):
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
    
# 4. Load the Stooq price data for the tickers found in the SEC filings.
if not os.path.exists("stooq_prices_2010_2025.parquet"):
    loader = StockPricesLoader()
    loader.load_stock_prices(
        sec_filings_parquet = "parsed_sec_filings_with_tickers.parquet",
        stooq_folder = "stooq_prices",
        # take from 2010 to have some history before our SEC filings data starts, to be able to compute features like 200-day moving average at the start of our SEC filings data in 2011.
        start_date = "2010-01-01",
        end_date = "2025-12-31",
        output_parquet = "stooq_prices_2010_2025.parquet"
    )

# 5. Merge SEC filings data with stock price data on ticker and date to create the final dataset.
if not os.path.exists("sec_filings_with_price_features_and_labels.parquet"):
    pricesSecFilingsCombiner = PriceSecFilingsCombiner()
    
    pricesSecFilingsCombiner.build_dataset(
        prices_path = "stooq_prices_2010_2025.parquet",
        filings_path = "parsed_sec_filings_with_tickers.parquet",
        output_path = "sec_filings_with_price_features_and_labels.parquet"
    )

# 6. Extract sentiment scores using FinBert from the text columns to create a complete numeric dataset.
if not os.path.exists("us_market_dataset.parquet"):    
    enhancer = FinBertSentimentEnhancer(
        chunk_size=384,
        stride=64,
        max_chunks=48,
        batch_size=512,         # GPU batch size
        doc_batch_size=512,     # docs per outer batch
        save_every=100000,
        cache_dir="finbert_cache",
        use_chunk_cache=False 
    )

    enhancer.add_sentiment_scores(
        input_path="sec_filings_with_price_features_and_labels.parquet",
        output_path="us_market_dataset.parquet"
    )