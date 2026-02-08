import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import time

class PriceSecFilingsCombiner:
    """
    Combines SEC filings data with stock price data to create a dataset of events.
    Each event is a filing aligned to the next trading day, with features extracted from past prices and labels for future returns.
    """
    def __init__(self, 
                lookback_days=60, # feature window size
                label_horizons=list(range(1, 8)), # 1..7 days ahead
                class_threshold=0.005  # 0.5% threshold for UP/DOWN classification
        ):
            self.lookback_days = lookback_days
            self.label_horizons = label_horizons
            self.class_threshold = class_threshold
    
    def build_dataset(self, prices_path, filings_path, output_path):
        print("Loading prices parquet...")
        prices = pd.read_parquet(prices_path)

        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.dropna(subset=["date", "ticker"])
        prices["ticker"] = prices["ticker"].astype(str).str.upper()

        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Normalize column names
        prices = prices.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        print("Loading filings parquet...")
        filings = pd.read_parquet(filings_path)

        filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
        filings = filings.dropna(subset=["filing_date", "ticker"])

        filings["ticker"] = filings["ticker"].astype(str).str.upper()
        filings["acceptance_time"] = filings["acceptance_time"].apply(self._parse_acceptance_time)

        # Group prices by ticker for fast lookup
        print("Grouping prices by ticker...")
        prices_by_ticker = {}
        trading_days_by_ticker = {}

        for ticker, df_t in prices.groupby("ticker"):
            df_t = df_t.sort_values("date").copy()
            df_t = df_t.set_index("date")
            prices_by_ticker[ticker] = df_t
            trading_days_by_ticker[ticker] = df_t.index

        print("Building event dataset...")
        rows = []

        for _, row in tqdm(filings.iterrows(), total=len(filings), desc="Aligning filings"):
            ticker = row["ticker"]

            if ticker not in prices_by_ticker:
                continue

            price_df = prices_by_ticker[ticker]
            trading_days = trading_days_by_ticker[ticker]

            filing_date = row["filing_date"]
            acceptance_time = row["acceptance_time"]

            event_day = self._get_event_trading_day(filing_date, acceptance_time, trading_days)

            if event_day is None:
                continue

            # Extract features from past prices
            features = self._compute_features(price_df, event_day, lookback=self.lookback_days)
            if features is None:
                continue

            # Compute labels (future returns + classification)
            labels = self._compute_labels(price_df, event_day, self.label_horizons, threshold=self.class_threshold)
            if labels is None:
                continue

            combined = row.to_dict()
            combined["event_day"] = event_day

            combined.update(features)
            combined.update(labels)

            rows.append(combined)

        final_df = pd.DataFrame(rows)

        if final_df.empty:
            print("No usable aligned samples were created.")
            return final_df

        final_df = final_df.sort_values(["ticker", "event_day"]).reset_index(drop=True)

        print(f"Final dataset size: {len(final_df):,} rows")
        print(f"Saving to parquet: {output_path}")

        final_df.to_parquet(output_path, index=False)
        
        return final_df

    def _parse_acceptance_time(self, x):
        """
        acceptance_time may be stored as:
        - "16:19:06"
        - pandas Timestamp
        - NaN
        Returns python datetime.time or None.
        """
        if pd.isna(x):
            return None

        if isinstance(x, pd.Timestamp):
            return x.time()

        if isinstance(x, str):
            try:
                parts = x.strip().split(":")
                return time(int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return None

        return None


    def _get_event_trading_day(self, filing_date, acceptance_time, trading_days):
        """
        filing_date: pandas.Timestamp normalized to date
        acceptance_time: datetime.time or None
        trading_days: sorted DatetimeIndex of available trading days for ticker

        Rule:
        - If acceptance_time >= 16:00:00 => next trading day
        - Else => same day if it's trading day, otherwise next trading day
        """
        filing_date = pd.Timestamp(filing_date).normalize()

        market_close = time(16, 0, 0)

        # find the first trading day >= filing_date
        idx = trading_days.searchsorted(filing_date)

        if idx >= len(trading_days):
            return None

        candidate_day = trading_days[idx]

        # If filing is after market close, shift by +1 trading day
        if acceptance_time is not None and acceptance_time >= market_close:
            if idx + 1 >= len(trading_days):
                return None
            return trading_days[idx + 1]

        return candidate_day

    def _compute_features(self, price_df, event_day, lookback=60):
        """
        price_df must be sorted by date, indexed by date.
        Uses historical window strictly BEFORE event_day.
        If insufficient history, uses as much as available.
        """
        if event_day not in price_df.index:
            return None

        t = price_df.index.get_loc(event_day)

        if t < 5:
            return None

        start = max(0, t - lookback)
        hist = price_df.iloc[start:t].copy()

        if len(hist) < 10:
            return None

        close = hist["close"].astype(float)
        volume = hist["volume"].astype(float)

        returns = close.pct_change().dropna()
        log_returns = np.log(close).diff().dropna()

        last_close = float(close.iloc[-1])

        # Moving averages (fallback is automatic since hist may be shorter)
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])

        # Expanding averages (always available)
        exp_mean = float(close.expanding().mean().iloc[-1])

        # Momentum
        momentum_5 = float((close.iloc[-1] / close.iloc[-6]) - 1) if len(close) >= 6 else np.nan
        momentum_20 = float((close.iloc[-1] / close.iloc[-21]) - 1) if len(close) >= 21 else np.nan

        # Volatility
        vol_5 = float(returns.tail(5).std()) if len(returns) >= 5 else np.nan
        vol_20 = float(returns.tail(20).std()) if len(returns) >= 20 else np.nan
        vol_full = float(returns.std()) if len(returns) > 2 else np.nan

        # Volume zscore (last volume vs window)
        vol_mean = float(volume.mean())
        vol_std = float(volume.std())
        volume_z = float((volume.iloc[-1] - vol_mean) / vol_std) if vol_std > 0 else np.nan

        # High-low range (last day in history)
        last_high = float(hist["high"].iloc[-1])
        last_low = float(hist["low"].iloc[-1])
        hl_range = float((last_high - last_low) / last_close) if last_close > 0 else np.nan

        # Relative to moving averages
        rel_ma20 = float((last_close / ma20) - 1) if ma20 > 0 else np.nan
        rel_ma50 = float((last_close / ma50) - 1) if ma50 > 0 else np.nan

        return {
            "hist_days_used": len(hist),
            "close_last": last_close,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma50": ma50,
            "expanding_mean": exp_mean,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "volatility_5": vol_5,
            "volatility_20": vol_20,
            "volatility_full": vol_full,
            "volume_zscore": volume_z,
            "hl_range": hl_range,
            "rel_ma20": rel_ma20,
            "rel_ma50": rel_ma50,
        }


    def _compute_labels(self, price_df, event_day, horizons, threshold=0.005):
        """
        event_day must exist in price_df index.
        Uses close-to-close returns:
            return_h = (close[t+h] - close[t]) / close[t]
        Classification:
            DOWN if < -threshold
            STAY if between
            UP if > threshold
        """
        
        def safe_pct_change(a, b):
            """Return (b-a)/a safely."""
            if a is None or b is None:
                return np.nan
            if a == 0:
                return np.nan
            return (b - a) / a
        
        if event_day not in price_df.index:
            return None

        t = price_df.index.get_loc(event_day)

        close_t = float(price_df.iloc[t]["close"])
        if close_t <= 0:
            return None

        out = {}

        for h in horizons:
            if t + h >= len(price_df):
                out[f"return_{h}d"] = np.nan
                out[f"label_{h}d"] = np.nan
                continue

            close_future = float(price_df.iloc[t + h]["close"])
            r = safe_pct_change(close_t, close_future)

            out[f"return_{h}d"] = r

            if np.isnan(r):
                out[f"label_{h}d"] = np.nan
            elif r > threshold:
                out[f"label_{h}d"] = 2   # UP
            elif r < -threshold:
                out[f"label_{h}d"] = 0   # DOWN
            else:
                out[f"label_{h}d"] = 1   # STAY

        return out

if __name__ == "__main__":
    pricesSecFilingsCombiner = PriceSecFilingsCombiner()
    
    pricesSecFilingsCombiner.build_event_dataset(
        prices_path = "stooq_prices_2010_2025.parquet",
        filings_path = "parsed_sec_filings_with_tickers_and_time.parquet",
        output_path = "sec_filings_with_price_features_and_labels.parquet"
    )