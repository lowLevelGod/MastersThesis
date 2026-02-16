import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def build_feature_matrix(df, drop_cols):
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X


def shuffle_sentiment_features(X, sentiment_cols, seed=42):
    """
    Shuffles sentiment columns across rows (independently),
    breaking the relationship between sentiment and target.
    """
    X_shuffled = X.copy()
    rng = np.random.default_rng(seed)

    for col in sentiment_cols:
        if col in X_shuffled.columns:
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)

    return X_shuffled


def logistic_regression_cv_test(X, y, tscv, C):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=4000, C=C))
    ])

    accs, f1s = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro"))

    return float(np.mean(accs)), float(np.mean(f1s))


# ============================================================
# MAIN SCRIPT
# ============================================================
df = pd.read_parquet("us_market_dataset.parquet")
df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
df = df.dropna(subset=["filing_date"]).sort_values("filing_date").reset_index(drop=True)

DATE_COL = "filing_date"

SENTIMENT_COLS = [
    "finbert_neg",
    "finbert_neu",
    "finbert_pos",
    "finbert_neg_std",
    "finbert_neu_std",
    "finbert_pos_std",
    "finbert_polarity_std"
]

NON_FEATURE_COLS = [
    "filing_date",
    "form_type",
    "cik",
    "accession",
    "ticker",
    "acceptance_time",
    "event_day",
    "event_base_price",
    "event_next_price",
    "event_return",
    "event_day_used",
    "event_label"
]

CLASSIFICATION_LABEL_COLS = [f"label_{day}d" for day in range(1, 8)]
REGRESSION_LABEL_COLS = [f"return_{day}d" for day in range(1, 8)]

DROP_COLS = NON_FEATURE_COLS + CLASSIFICATION_LABEL_COLS + REGRESSION_LABEL_COLS


# Build X
X = build_feature_matrix(df, DROP_COLS)

# Setup CV
tscv = TimeSeriesSplit(n_splits=5)

def repeated_shuffle_test(X, y, tscv, sentiment_cols, n_trials=20, C=1):
    print(f"Number of trials: {n_trials}, C: {C}")
    real_acc, real_f1 = logistic_regression_cv_test(X, y, tscv, C=C)

    shuffled_f1s = []
    for trial in tqdm(range(n_trials)):
        X_shuf = shuffle_sentiment_features(X, sentiment_cols, seed=trial * 999)
        _, shuf_f1 = logistic_regression_cv_test(X_shuf, y, tscv, C=C)
        shuffled_f1s.append(shuf_f1)

    shuffled_f1s = np.array(shuffled_f1s)

    return {
        "real_f1": real_f1,
        "shuffled_mean_f1": float(shuffled_f1s.mean()),
        "shuffled_std_f1": float(shuffled_f1s.std()),
        "delta_real_minus_mean": float(real_f1 - shuffled_f1s.mean()),
        "p_value_like": float(np.mean(shuffled_f1s >= real_f1))  # fraction where shuffled >= real
    }


print("\n================ REPEATED SHUFFLE TEST (LOGREG) ================\n")

# pick C values based on grid search for each horizon day.
for (day, C) in zip(range(1, 8), [0.1, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0]):
    label_col = f"label_{day}d"
    y = df[label_col].astype(int)

    stats = repeated_shuffle_test(X, y, tscv, SENTIMENT_COLS, n_trials=1000, C=C)

    print(f"--- Horizon {day} Day(s) ---")
    print(f"Real F1:             {stats['real_f1']:.4f}")
    print(f"Shuffled Mean F1:    {stats['shuffled_mean_f1']:.4f}")
    print(f"Shuffled Std F1:     {stats['shuffled_std_f1']:.4f}")
    print(f"Δ (Real - ShufMean): {stats['delta_real_minus_mean']:+.4f}")
    print(f"Fraction shuffled >= real (p-like): {stats['p_value_like']:.4f}")
    print()