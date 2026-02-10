from sklearn.model_selection import TimeSeriesSplit

class ExpandingWindowSplitter:
    def __init__(self, n_splits=5, test_ratio=0.2):
        self.n_splits = n_splits
        self.test_ratio = test_ratio

    def split(self, data):
        n = len(data)
        test_size = int(n * self.test_ratio)

        train_val = data.iloc[:-test_size]
        test = data.iloc[-test_size:]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        return train_val, test, tscv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def cross_validate(self, X, y, splitter):
        accs, f1s = [], []

        for train_idx, val_idx in splitter.split(X):
            self.pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = self.pipeline.predict(X.iloc[val_idx])

            accs.append(accuracy_score(y.iloc[val_idx], preds))
            f1s.append(f1_score(y.iloc[val_idx], preds))

        return np.mean(accs), np.mean(f1s)

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def evaluate(self, X, y):
        preds = self.pipeline.predict(X)
        return accuracy_score(y, preds), f1_score(y, preds) 

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class FinBERTSentiment:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score(self, text: str):
        if not text or text.strip() == "":
            return {
                "sent_pos": 0.0,
                "sent_neg": 0.0,
                "sent_neu": 0.0
            }

        inputs = self.tokenizer(
            text[:4000], 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy()[0]

        return {
            "sent_pos": float(probs[0]),
            "sent_neg": float(probs[1]),
            "sent_neu": float(probs[2])
        }
    
import pandas as pd

def enhance_sec_filings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["filing_date"])

    finbert = FinBERTSentiment()

    enhanced_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        new_row = row.to_dict()

        for section in ["risk_factors_text", "management_discussion_text", "market_risk_text"]:
            text_col = f"{section}_text"

            text = row.get(text_col, "")
            missing = int(not isinstance(text, str) or text.strip() == "")
            new_row[f"{section}_missing"] = missing

            scores = finbert.score(text)
            for k, v in scores.items():
                new_row[f"{section}_{k}"] = v

        enhanced_rows.append(new_row)

    return pd.DataFrame(enhanced_rows)

import yfinance as yf

class PriceLoader:
    def __init__(self, start="2000-01-01", end="2026-01-01"):
        self.start = start
        self.end = end
        self.cache = {}

    def get(self, ticker):
        if ticker not in self.cache:
            df = yf.download(ticker, start=self.start, end=self.end)
            df.index = pd.to_datetime(df.index)
            self.cache[ticker] = df
        return self.cache[ticker]

def build_multi_ticker_dataset(sec_df: pd.DataFrame) -> pd.DataFrame:
    price_loader = PriceLoader()

    feature_extractor = PriceFeatureExtractor(window=20)
    label_generator = LabelGenerator(horizon=1)

    builder = EventDatasetBuilder(
        feature_extractor=feature_extractor,
        label_generator=label_generator
    )

    all_rows = []

    for ticker, group in sec_df.groupby("ticker"):
        prices = price_loader.get(ticker)

        if prices is None or prices.empty:
            continue

        group = group.rename(columns={"filing_date": "event_day"})
        group = group.sort_values("event_day")

        try:
            df_ticker = builder.build(prices, group)
            all_rows.append(df_ticker)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    return pd.concat(all_rows, ignore_index=True)

enhanced_sec = enhance_sec_filings("sec_filings.csv")
enhanced_sec.to_csv("sec_filings_enhanced.csv", index=False)

sec_df = pd.read_csv(
    "sec_filings_enhanced.csv",
    parse_dates=["filing_date"]
)

final_dataset = build_multi_ticker_dataset(sec_df)

final_dataset.to_csv("final_model_dataset.csv", index=False)

df = pd.read_csv(
    "final_model_dataset.csv",
    parse_dates=["event_day"]
).sort_values("event_day")

TARGET = "label"

DROP_COLS = [
    "ticker",
    "form_type",
    "event_day",
    "risk_factors_text",
    "management_discussion_text",
    "market_risk_text",
    TARGET
]

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

print(X.head())

splitter = ExpandingWindowSplitter(n_splits=5, test_ratio=0.2)
train_val, test, tscv = splitter.split(df)

X_train_val = X.loc[train_val.index]
y_train_val = y.loc[train_val.index]

trainer = ModelTrainer()
cv_acc, cv_f1 = trainer.cross_validate(X_train_val, y_train_val, tscv)

print(f"CV Accuracy: {cv_acc:.3f}, CV F1: {cv_f1:.3f}")

trainer.fit(X_train_val, y_train_val)

X_test = X.loc[test.index]
y_test = y.loc[test.index]

test_acc, test_f1 = trainer.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}, Test F1: {test_f1:.3f}")