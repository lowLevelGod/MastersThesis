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
                class_threshold=0.01  # 1% threshold for UP/DOWN classification
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

        print("Loading filings parquet...")
        filings = pd.read_parquet(filings_path)

        filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
        filings = filings.dropna(subset=["filing_date", "ticker"])
        filings["ticker"] = filings["ticker"].astype(str).str.upper()

        def parse_acceptance_time(x):
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

        filings["acceptance_time"] = filings["acceptance_time"].apply(parse_acceptance_time)

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

            filing_date = pd.Timestamp(row["filing_date"]).normalize()
            acceptance_time = row["acceptance_time"]
            
            def get_event_trading_day(filing_date, trading_days):
                """
                Returns the first trading day >= filing_date.
                No shifting based on acceptance_time, since labels handle that.
                """
                filing_date = pd.Timestamp(filing_date).normalize()

                idx = trading_days.searchsorted(filing_date)

                if idx >= len(trading_days):
                    return None

                return trading_days[idx]

            event_day = get_event_trading_day(filing_date, trading_days)
            if event_day is None:
                continue

            features = self._compute_features(price_df, event_day, lookback=self.lookback_days)
            if features is None:
                continue

            labels = self._compute_labels(
                price_df,
                event_day,
                acceptance_time,
                self.label_horizons,
                self.class_threshold
            )

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

        final_df["event_day"] = pd.to_datetime(final_df["event_day"], errors="coerce")
        final_df = final_df.dropna(subset=["event_day"])

        final_df = final_df.sort_values(["ticker", "event_day"]).reset_index(drop=True)

        print(f"Final dataset size: {len(final_df):,} rows")
        print(f"Saving to parquet: {output_path}")

        final_df.to_parquet(output_path, index=False)

        return final_df

    def _get_event_price_points(self, price_df, filing_date, acceptance_time):
        """
        Returns the correct (base_price, next_price) pair depending on filing time.

        price_df: indexed by trading day (DatetimeIndex), must contain open/close
        filing_date: Timestamp normalized to date
        acceptance_time: datetime.time or None

        Output:
            (base_price, next_price, label_day_used)

        label_day_used = the trading day used for the main event (filing_date aligned)
        """

        trading_days = price_df.index
        filing_date = pd.Timestamp(filing_date).normalize()

        # if no acceptance time, assume after close (most conservative)
        if acceptance_time is None:
            acceptance_time = time(16, 1, 0)

        market_open = time(9, 30, 0)
        market_close = time(16, 0, 0)

        idx = trading_days.searchsorted(filing_date)

        if idx >= len(trading_days):
            return None

        day = trading_days[idx]

        # if filing_date is not a trading day, day will be next trading day
        # treat it as "before open" on that next trading day
        if day != filing_date:
            if idx == 0:
                return None
            prev_day = trading_days[idx - 1]
            base_price = float(price_df.loc[prev_day, "close"])
            next_price = float(price_df.loc[day, "open"])
            return base_price, next_price, day

        # filing_date is a trading day
        if acceptance_time < market_open:
            # before open: prev close -> today's open
            if idx == 0:
                return None
            prev_day = trading_days[idx - 1]
            base_price = float(price_df.loc[prev_day, "close"])
            next_price = float(price_df.loc[day, "open"])
            return base_price, next_price, day

        elif acceptance_time <= market_close:
            # during hours: today's open -> today's close
            base_price = float(price_df.loc[day, "open"])
            next_price = float(price_df.loc[day, "close"])
            return base_price, next_price, day

        else:
            # after close: today's close -> next trading day open
            if idx + 1 >= len(trading_days):
                return None
            next_day = trading_days[idx + 1]
            base_price = float(price_df.loc[day, "close"])
            next_price = float(price_df.loc[next_day, "open"])
            return base_price, next_price, day

    def _compute_features(self, price_df, event_day, lookback=60):
        """
        price_df must be sorted by date, indexed by date.
        Uses historical window strictly BEFORE event_day.
        Always returns valid floats (no NaNs / inf) using fallbacks.
        """

        if event_day not in price_df.index:
            return None

        t = price_df.index.get_loc(event_day)

        # need at least 2 historical points to compute anything meaningful
        if t < 2:
            return None

        start = max(0, t - lookback)
        hist = price_df.iloc[start:t].copy()

        if len(hist) < 2:
            return None

        # -----------------------------
        # Helpers
        # -----------------------------
        def safe_float(x, default=0.0):
            """Convert to float and guarantee finite output."""
            try:
                x = float(x)
                if not np.isfinite(x):
                    return default
                return x
            except Exception:
                return default

        def clean_series(series):
            """Convert to float, drop NaNs/infs."""
            s = pd.to_numeric(series, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return s.astype(float)

        def rolling_or_expanding_mean(series, window):
            """If not enough history for rolling mean, fallback to expanding mean."""
            if len(series) == 0:
                return 0.0

            if len(series) >= window:
                val = series.rolling(window).mean().iloc[-1]
            else:
                val = series.expanding().mean().iloc[-1]

            return safe_float(val, default=safe_float(series.iloc[-1], 0.0))

        def safe_momentum(series, target_lag):
            """
            Compute momentum using as many days as available.
            If not enough for target_lag, use max possible lag.
            """
            if len(series) < 2:
                return 0.0

            lag = min(target_lag, len(series) - 1)
            prev_price = safe_float(series.iloc[-(lag + 1)], 0.0)
            last_price = safe_float(series.iloc[-1], 0.0)

            if prev_price <= 0:
                return 0.0

            return safe_float((last_price / prev_price) - 1, 0.0)

        def safe_std(arr):
            """Safe standard deviation (returns 0 if not enough samples)."""
            if arr is None or len(arr) < 2:
                return 0.0

            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]

            if len(arr) < 2:
                return 0.0

            s = np.std(arr, ddof=1)
            return safe_float(s, 0.0)

        # -----------------------------
        # Clean core columns
        # -----------------------------
        close = clean_series(hist["close"])
        volume = clean_series(hist["volume"])
        high = clean_series(hist["high"])
        low = clean_series(hist["low"])

        # If cleaning destroyed the history, bail
        if len(close) < 2 or len(volume) < 1:
            return None

        last_close = safe_float(close.iloc[-1], 0.0)

        # -----------------------------
        # Returns
        # -----------------------------
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        vol_5 = safe_std(returns[-5:])
        vol_20 = safe_std(returns[-20:])
        vol_full = safe_std(returns)

        # -----------------------------
        # Moving averages (with expanding fallback)
        # -----------------------------
        ma5 = rolling_or_expanding_mean(close, 5)
        ma10 = rolling_or_expanding_mean(close, 10)
        ma20 = rolling_or_expanding_mean(close, 20)
        ma50 = rolling_or_expanding_mean(close, 50)

        # -----------------------------
        # Momentum (variable lag fallback)
        # -----------------------------
        momentum_5 = safe_momentum(close, 5)
        momentum_20 = safe_momentum(close, 20)

        # -----------------------------
        # Volume z-score
        # -----------------------------
        vol_mean = safe_float(volume.mean(), 0.0)
        vol_std = safe_float(volume.std(ddof=1), 0.0)

        if vol_std > 0:
            volume_z = safe_float((volume.iloc[-1] - vol_mean) / vol_std, 0.0)
        else:
            volume_z = 0.0

        # -----------------------------
        # High-low range (last historical day)
        # -----------------------------
        # If high/low got cleaned and are shorter than close, use last available
        last_high = safe_float(high.iloc[-1], last_close)
        last_low = safe_float(low.iloc[-1], last_close)

        if last_close > 0:
            hl_range = safe_float((last_high - last_low) / last_close, 0.0)
        else:
            hl_range = 0.0

        return {
            "close_last": last_close,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma50": ma50,
            "momentum_5": momentum_5,
            "momentum_20": momentum_20,
            "volatility_5": vol_5,
            "volatility_20": vol_20,
            "volatility_full": vol_full,
            "volume_zscore": volume_z,
            "hl_range": hl_range,
        }

    def _compute_labels(
        self,
        price_df,
        filing_date,
        acceptance_time,
        horizons=(1,2,3,4,5,6,7),
        threshold=0.005
    ):
        """
        Creates returns + classification labels using acceptance_time logic.

        For horizon=1, it uses the *immediate next market reaction*:
            pre-market: prev close -> open
            during: open -> close
            after-hours: close -> next open

        For horizon > 1:
            it measures from the event "next_price" to future close prices.
        """

        def safe_pct_change(a, b):
            if a is None or b is None:
                return np.nan
            if a == 0:
                return np.nan
            return (b - a) / a

        if filing_date is None:
            return None

        res = self._get_event_price_points(price_df, filing_date, acceptance_time)
        if res is None:
            return None

        base_price, next_price, label_day = res

        if base_price <= 0 or next_price <= 0:
            return None

        out = {}

        # immediate reaction return (horizon 1 = event reaction)
        r0 = safe_pct_change(base_price, next_price)

        out["event_base_price"] = base_price
        out["event_next_price"] = next_price
        out["event_return"] = r0
        out["event_day_used"] = label_day

        # classification for immediate reaction
        if np.isnan(r0):
            out["event_label"] = np.nan
        elif r0 > threshold:
            out["event_label"] = 2
        elif r0 < -threshold:
            out["event_label"] = 0
        else:
            out["event_label"] = 1

        # Now compute future returns relative to next_price
        # event anchor day is label_day
        t = price_df.index.get_loc(label_day)

        for h in horizons:
            # for h=1: use the immediate reaction return
            if h == 1:
                out["return_1d"] = r0
                out["label_1d"] = out["event_label"]
                continue

            future_idx = t + (h - 1)
            if future_idx >= len(price_df):
                out[f"return_{h}d"] = np.nan
                out[f"label_{h}d"] = np.nan
                continue

            future_close = float(price_df.iloc[future_idx]["close"])
            r = safe_pct_change(next_price, future_close)

            out[f"return_{h}d"] = r

            if np.isnan(r):
                out[f"label_{h}d"] = np.nan
            elif r > threshold:
                out[f"label_{h}d"] = 2
            elif r < -threshold:
                out[f"label_{h}d"] = 0
            else:
                out[f"label_{h}d"] = 1

        return out

if __name__ == "__main__":
    pricesSecFilingsCombiner = PriceSecFilingsCombiner()
    
    pricesSecFilingsCombiner.build_dataset(
        prices_path = "stooq_prices_2010_2025.parquet",
        filings_path = "parsed_sec_filings_with_tickers_and_time.parquet",
        output_path = "sec_filings_with_price_features_and_labels.parquet"
    )