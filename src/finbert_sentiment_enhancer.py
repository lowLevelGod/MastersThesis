import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import os
import pickle

class FinBertSentimentEnhancer:
    TEXT_COLS = ["mda_text", "market_risk_text"]
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(
        self,
        chunk_size,
        stride,
        max_chunks,
        batch_size
    ):
        self.chunk_size = chunk_size
        self.stride = stride 
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        
    def add_sentiment_scores(
        self,
        input_path,
        output_path
    ):
        print("Loading dataset...")
        df = pd.read_parquet(input_path)

        print("Dataset shape:", df.shape)

        # Ensure missing text cols exist
        for col in self.TEXT_COLS:
            if col not in df.columns:
                df[col] = ""

        df = self._run_finbert_sentiment(df)

        print("Dropping raw text columns...")
        df = df.drop(columns=self.TEXT_COLS, errors="ignore")

        print("Saving final dataset to:", output_path)
        df.to_parquet(output_path, index=False)

        print("Done.")
        print(df[["finbert_neg", "finbert_neu", "finbert_pos"]].describe())

    def _chunk_cache_path(self, idx):
        return os.path.join("finbert_chunk_cache", f"doc_chunks_{idx}.pkl")

    def _chunk_text_stratified(self, text, tokenizer, chunk_size=None, stride=None, max_chunks=None, seed=42):
        """
        Split text into overlapping token chunks, sample chunks giving more weight to middle and end.
        """
        chunk_size = chunk_size or self.chunk_size
        stride = stride or self.stride
        max_chunks = max_chunks or self.max_chunks

        if not text:
            return []

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return []

        # build all overlapping chunks
        chunks = []
        start = 0
        step = chunk_size - stride

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            if end == len(tokens):
                break
            start += step

        if len(chunks) <= max_chunks:
            return chunks

        # stratified selection: more weight to middle+end
        random.seed(seed)

        n_head = max_chunks // 6      # smaller head fraction
        n_tail = max_chunks // 3      # bigger tail fraction
        n_mid = max_chunks - n_head - n_tail

        head = chunks[:n_head]
        tail = chunks[-n_tail:]
        middle = chunks[n_head:len(chunks)-n_tail]

        if len(middle) > n_mid:
            middle = random.sample(middle, n_mid)

        return head + middle + tail

    def _run_finbert_sentiment(self, df):
        print("Loading FinBERT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        model.to(device)
        model.eval()

        sent_neg = np.zeros(len(df), dtype=np.float32)
        sent_neu = np.zeros(len(df), dtype=np.float32)
        sent_pos = np.zeros(len(df), dtype=np.float32)

        if not os.path.exists("finbert_chunk_cache/"):
            os.makedirs("finbert_chunk_cache/")
            
        print("Preparing or loading document chunks...")
        all_doc_chunks = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing docs"):
            cache_path = self._chunk_cache_path(i)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    chunks = pickle.load(f)
            else:
                mda = row.get("mda_text", "")
                risk = row.get("market_risk_text", "")
                combined = (mda + "\n\n" + risk).strip()

                chunks = self._chunk_text_stratified(
                    combined, tokenizer,
                    chunk_size=self.chunk_size,
                    stride=self.stride,
                    max_chunks=self.max_chunks
                )

                # save for later
                with open(cache_path, "wb") as f:
                    pickle.dump(chunks, f)

            all_doc_chunks.append(chunks)

        print("Running FinBERT inference...")
        with torch.no_grad():
            for i in tqdm(range(len(df)), desc="FinBERT scoring"):
                chunks = all_doc_chunks[i]

                if len(chunks) == 0:
                    sent_neg[i], sent_neu[i], sent_pos[i] = 0.0, 1.0, 0.0
                    continue

                chunk_probs = []
                for j in range(0, len(chunks), self.batch_size):
                    batch_chunks = chunks[j:j+self.batch_size]
                    encoded = tokenizer.pad({"input_ids": batch_chunks}, padding=True, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    chunk_probs.extend(probs)

                agg = self._aggregate_chunk_probs(chunk_probs)
                sent_neg[i], sent_neu[i], sent_pos[i] = map(float, agg)

        df["finbert_neg"] = sent_neg
        df["finbert_neu"] = sent_neu
        df["finbert_pos"] = sent_pos

        return df

    def _aggregate_chunk_probs(self, chunk_probs):
        """
        chunk_probs: list of np arrays [neg, neu, pos]
        Return averaged probs.
        """
        if len(chunk_probs) == 0:
            return np.array([0.0, 1.0, 0.0])  # default neutral

        chunk_probs = np.vstack(chunk_probs)
        avg = chunk_probs.mean(axis=0)
        return avg

if __name__ == "__main__":
    finBertSentimentEnhancer = FinBertSentimentEnhancer(
        chunk_size = 384,
        stride = 64,             
        max_chunks = 20,
        batch_size = 32 
    )
    
    finBertSentimentEnhancer.add_sentiment_scores(
        input_path = "sec_filings_with_price_features_and_labels.parquet",
        output_path = "final_dataset.parquet"
    )