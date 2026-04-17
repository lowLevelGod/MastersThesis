import os
import random
import pickle
import gc
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


class FinBertSentimentEnhancer:
    TEXT_COLS = ["mda_text", "market_risk_text"]
    MODEL_NAME = "yiyanghkust/finbert-tone" # this was pretrained on 10-K & 10-Q filings

    def __init__(
        self,
        chunk_size=384,
        stride=64,
        max_chunks=20,
        batch_size=64,          # chunk batch size for GPU
        doc_batch_size=32,      # how many documents to chunk + infer together
        save_every=200,
        cache_dir="finbert_cache",
        use_chunk_cache=True,   # set False on Kaggle for speed
        seed=42,
        shard_id=0,    
        num_shards=1,         
    ):
        self.chunk_size = chunk_size
        self.stride = stride
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.doc_batch_size = doc_batch_size
        self.save_every = save_every
        self.cache_dir = cache_dir
        self.use_chunk_cache = use_chunk_cache
        self.seed = seed
        
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.chunk_cache_dir = os.path.join(cache_dir, "chunk_cache")
        self.result_cache_path = os.path.join(
            cache_dir,
            f"finbert_results_checkpoint_shard_{self.shard_id}.parquet"
        )
        
        os.makedirs(self.cache_dir, exist_ok=True)
        # os.makedirs(self.chunk_cache_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        print("Loading FinBERT self.model...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=3)
        
        self.model.to(self.device)
        
        self.id2label = self.model.config.id2label
        self.label2id = {v.lower(): k for k, v in self.id2label.items()}

        print("Model label mapping:", self.id2label)
        
        try:
            self.neg_idx = self.label2id["negative"]
            self.neu_idx = self.label2id["neutral"]
            self.pos_idx = self.label2id["positive"]
        except KeyError:
            raise ValueError(
                f"Model labels must include 'negative', 'neutral', 'positive'. "
                f"Found: {list(self.label2id.keys())}"
            )

        print("Internal reorder indices:",
            "neg=", self.neg_idx,
            "neu=", self.neu_idx,
            "pos=", self.pos_idx)

    # -----------------------------
    # Public API
    # -----------------------------
    def add_sentiment_scores(self, input_path, output_path=None):
        print("Loading dataset...")
        df = pd.read_parquet(input_path)
        print("Dataset shape:", df.shape)

        for col in self.TEXT_COLS:
            if col not in df.columns:
                df[col] = ""

        df = df.reset_index(drop=True)

        df = self._run_finbert_sentiment_batched(df)

        print("Dropping raw text columns...")
        df = df.drop(columns=self.TEXT_COLS, errors="ignore")

        print("Saving final dataset to:", output_path)
        if output_path is not None:
            df.to_parquet(output_path, index=False)

        print("Done.")
        print(df[
            [
                "finbert_neg", "finbert_neu", "finbert_pos",
                "finbert_neg_std", "finbert_neu_std", "finbert_pos_std",
                "finbert_polarity_std"
            ]
        ].describe())

    # -----------------------------
    # Chunk caching helpers
    # -----------------------------
    def _chunk_cache_path(self, idx):
        return os.path.join(self.chunk_cache_dir, f"doc_chunks_{idx}.pkl")

    def _load_or_create_chunks(self, idx, row):
        """
        Loads chunks from disk if caching enabled, otherwise generates them.
        """
        cache_path = self._chunk_cache_path(idx)

        if self.use_chunk_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        mda = row.get("mda_text", "") or ""
        risk = row.get("market_risk_text", "") or ""
        combined = (mda + "\n\n" + risk).strip()

        chunks = self._chunk_text_stratified(combined)

        if self.use_chunk_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(chunks, f)
            except Exception:
                pass

        return chunks

    # -----------------------------
    # Chunking strategy
    # -----------------------------
    def _chunk_text_stratified(self, text):
        """
        Chunk the full document into overlapping token windows.
        Stratified selection:
          - small fraction head
          - medium random middle
          - larger tail
        This biases toward middle+end (your important info region).
        """
        if not text:
            return []

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return []

        chunks = []
        start = 0
        step = self.chunk_size - self.stride

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            if end == len(tokens):
                break
            start += step

        if len(chunks) <= self.max_chunks:
            return chunks

        random.seed(self.seed)

        # bias toward tail + middle
        n_head = max(1, self.max_chunks // 8)
        n_tail = max(1, self.max_chunks // 2)
        n_mid = self.max_chunks - n_head - n_tail

        head = chunks[:n_head]
        tail = chunks[-n_tail:]

        middle = chunks[n_head:len(chunks) - n_tail]
        if len(middle) > n_mid:
            middle = random.sample(middle, n_mid)

        # shuffle middle so we don't always pick early-middle
        random.shuffle(middle)

        return head + middle + tail

    # -----------------------------
    # Checkpointing
    # -----------------------------
    def _ensure_sentiment_columns(self, df):
        needed_cols = [
            "finbert_neg", "finbert_neu", "finbert_pos",
            "finbert_neg_std", "finbert_neu_std", "finbert_pos_std",
            "finbert_polarity_std"
        ]
        for c in needed_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def _load_checkpoint(self, df):
        if os.path.exists(self.result_cache_path):
            print(f"Found checkpoint: {self.result_cache_path}")
            ckpt = pd.read_parquet(self.result_cache_path)

            # for col in ckpt.columns:
            #     if col in df.columns and len(ckpt[col]) == len(df[col]):
            #         df[col] = ckpt[col].values
            self._ensure_sentiment_columns(df)
            df = ckpt
            print("Checkpoint loaded.")
        else:
            df = self._ensure_sentiment_columns(df)
            print("No checkpoint found. Starting fresh.")

        return df

    def _save_checkpoint(self, df):
        save_df = df.drop(columns=self.TEXT_COLS, errors="ignore")
        save_df.to_parquet(self.result_cache_path, index=False)

    # -----------------------------
    # Aggregation
    # -----------------------------
    def _aggregate_chunk_probs(self, chunk_probs):
        """
        chunk_probs: list of [neg, neu, pos]

        Returns:
          mean_probs: [neg, neu, pos]
          std_probs:  [neg, neu, pos]
          polarity_std: std(pos-neg)
        """
        if len(chunk_probs) == 0:
            return (
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                0.0
            )

        arr = np.vstack(chunk_probs).astype(np.float32)

        mean_probs = arr.mean(axis=0)
        std_probs = arr.std(axis=0, ddof=0)

        polarity = arr[:, 2] - arr[:, 0]  # pos - neg
        polarity_std = float(np.std(polarity, ddof=0))

        return mean_probs, std_probs, polarity_std
    
    # -----------------------------
    # Main batched inference
    # -----------------------------
    def _run_finbert_sentiment_batched(self, df):
        self.model.eval()

        checkpoint_df = self._load_checkpoint(df)

        all_shard_idxs = self._get_shard_indices(checkpoint_df)

        unfinished_mask = checkpoint_df["finbert_neg"].isna().values
        unfinished_idxs = [
            idx for idx in all_shard_idxs
            if unfinished_mask[idx]
        ]

        unfinished_idxs = np.array(unfinished_idxs)

        print(f"Total docs: {len(checkpoint_df)}")
        print(f"Remaining docs: {len(unfinished_idxs)}")

        if len(unfinished_idxs) == 0:
            print("All docs already processed.")
            return df

        processed_since_save = 0

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(unfinished_idxs), self.doc_batch_size),
                desc="Doc batches"
            ):
                batch_idxs = unfinished_idxs[batch_start:batch_start + self.doc_batch_size]

                all_chunks = []
                chunk_owner = []

                probs_by_doc = {idx: [] for idx in batch_idxs}

                # -----------------------------
                # Chunk all docs in batch
                # -----------------------------
                for idx in batch_idxs:
                    row = df.iloc[idx]
                    chunks = self._load_or_create_chunks(idx, row)

                    if len(chunks) == 0:
                        df.at[idx, "finbert_neg"] = 0.0
                        df.at[idx, "finbert_neu"] = 1.0
                        df.at[idx, "finbert_pos"] = 0.0

                        df.at[idx, "finbert_neg_std"] = 0.0
                        df.at[idx, "finbert_neu_std"] = 0.0
                        df.at[idx, "finbert_pos_std"] = 0.0
                        df.at[idx, "finbert_polarity_std"] = 0.0

                        processed_since_save += 1
                        continue

                    for c in chunks:
                        all_chunks.append(c)
                        chunk_owner.append(idx)

                if len(all_chunks) == 0:
                    continue

                # -----------------------------
                # Run inference on ALL chunks
                # -----------------------------
                for j in range(0, len(all_chunks), self.batch_size):
                    batch_chunks = all_chunks[j:j + self.batch_size]
                    batch_owners = chunk_owner[j:j + self.batch_size]

                    encoded = self.tokenizer.pad(
                        {"input_ids": batch_chunks},
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)

                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                    for p, owner in zip(probs, batch_owners):
                        # Dynamically reorder into [neg, neu, pos]
                        reordered = np.array([
                            p[self.neg_idx],
                            p[self.neu_idx],
                            p[self.pos_idx]
                        ], dtype=np.float32)

                        probs_by_doc[owner].append(reordered)

                    # free memory
                    del encoded, input_ids, attention_mask, logits, probs
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                # -----------------------------
                # Aggregate per doc
                # -----------------------------
                for idx in batch_idxs:
                    if not np.isnan(df.at[idx, "finbert_neg"]):
                        continue

                    mean_probs, std_probs, polarity_std = self._aggregate_chunk_probs(probs_by_doc[idx])

                    df.at[idx, "finbert_neg"] = float(mean_probs[0])
                    df.at[idx, "finbert_neu"] = float(mean_probs[1])
                    df.at[idx, "finbert_pos"] = float(mean_probs[2])

                    df.at[idx, "finbert_neg_std"] = float(std_probs[0])
                    df.at[idx, "finbert_neu_std"] = float(std_probs[1])
                    df.at[idx, "finbert_pos_std"] = float(std_probs[2])

                    df.at[idx, "finbert_polarity_std"] = float(polarity_std)

                    processed_since_save += 1

                # cleanup big lists (important for Kaggle RAM)
                del all_chunks, chunk_owner, probs_by_doc
                gc.collect()

                # -----------------------------
                # Checkpoint save
                # -----------------------------
                if processed_since_save >= self.save_every:
                    print(f"\nSaving checkpoint after {processed_since_save} processed docs...")
                    self._save_checkpoint(df)
                    processed_since_save = 0

            print("Saving final checkpoint...")
            self._save_checkpoint(df)

        return df
    
    def _get_shard_indices(self, df):
        total_indices = np.arange(len(df))

        # Otherwise split evenly by shard_id
        shard_indices = total_indices[
            total_indices % self.num_shards == self.shard_id
        ]

        return shard_indices

if __name__ == "__main__":
    
    shard_id = 0
    num_shards = 2

    enhancer = FinBertSentimentEnhancer(
        chunk_size=384,
        stride=64,
        max_chunks=48,
        batch_size=512,         # GPU batch size
        doc_batch_size=512,     # docs per outer batch
        save_every=5000,
        cache_dir="finbert_cache",
        use_chunk_cache=False,  # Kaggle: better False (disk I/O is slow)
        shard_id=shard_id,
        num_shards=num_shards
    )

    enhancer.add_sentiment_scores(
        input_path="/kaggle/input/sec-filings-with-stock-price-features/sec_filings_with_price_features_and_labels_shortened.parquet",
        output_path=f"us_market_dataset_shard_{shard_id}.parquet"
    )