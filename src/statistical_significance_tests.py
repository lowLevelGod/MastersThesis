import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, chi2

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss


# -----------------------------
# Settings
# -----------------------------
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

ALL_DROP_COLS = NON_FEATURE_COLS + CLASSIFICATION_LABEL_COLS + REGRESSION_LABEL_COLS


# -----------------------------
# Feature builder
# -----------------------------
def build_feature_matrix(df, drop_sentiment=False):
    X = df.drop(columns=ALL_DROP_COLS, errors="ignore")

    if drop_sentiment:
        X = X.drop(columns=SENTIMENT_COLS, errors="ignore")

    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X


# -----------------------------
# Logistic Regression pipeline
# -----------------------------
def make_logreg(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            C=C
        ))
    ])


# -----------------------------
# (A) Paired Fold Test
# -----------------------------
def paired_fold_test(X_sent, X_nosent, y, splitter, C=1.0):
    """
    For each fold:
      - train Sent model
      - train NoSent model
      - compute F1 macro
    Returns fold deltas + p-values.
    """
    sent_f1s = []
    nosent_f1s = []
    deltas = []

    for train_idx, val_idx in splitter.split(X_sent):
        X_train_sent, X_val_sent = X_sent.iloc[train_idx], X_sent.iloc[val_idx]
        X_train_ns, X_val_ns = X_nosent.iloc[train_idx], X_nosent.iloc[val_idx]

        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_sent = make_logreg(C=C)
        model_ns = make_logreg(C=C)

        model_sent.fit(X_train_sent, y_train)
        model_ns.fit(X_train_ns, y_train)

        pred_sent = model_sent.predict(X_val_sent)
        pred_ns = model_ns.predict(X_val_ns)

        f1_sent = f1_score(y_val, pred_sent, average="macro")
        f1_ns = f1_score(y_val, pred_ns, average="macro")

        sent_f1s.append(f1_sent)
        nosent_f1s.append(f1_ns)
        deltas.append(f1_sent - f1_ns)

    sent_f1s = np.array(sent_f1s)
    nosent_f1s = np.array(nosent_f1s)
    deltas = np.array(deltas)

    # paired t-test
    t_stat, t_p = ttest_rel(sent_f1s, nosent_f1s)

    # Wilcoxon signed rank (better for small folds)
    # if deltas are all zero, wilcoxon will error
    if np.allclose(deltas, 0):
        w_stat, w_p = np.nan, 1.0
    else:
        w_stat, w_p = wilcoxon(deltas)

    return {
        "sent_fold_f1": sent_f1s,
        "nosent_fold_f1": nosent_f1s,
        "delta_fold_f1": deltas,
        "mean_delta": float(np.mean(deltas)),
        "ttest_p": float(t_p),
        "wilcoxon_p": float(w_p)
    }


# -----------------------------
# (B) Block Bootstrap CI Test
# -----------------------------
def block_bootstrap_delta_f1(
    X_sent, X_nosent, y,
    test_idx,
    model_C=1.0,
    n_boot=500,
    block_size=200,
    seed=42
):
    """
    Block bootstrap on test set indices:
      - sample blocks of contiguous indices (time-series aware)
      - compute delta F1 for each bootstrap sample
      - return CI + p-like probability of delta <= 0
    """

    rng = np.random.default_rng(seed)

    test_idx = np.array(test_idx)
    n_test = len(test_idx)

    # build contiguous block starting points
    max_start = n_test - block_size
    if max_start <= 0:
        raise ValueError("block_size is too large for test set size")

    deltas = []

    for _ in range(n_boot):
        sampled_positions = []

        while len(sampled_positions) < n_test:
            start = rng.integers(0, max_start + 1)
            block = list(range(start, start + block_size))
            sampled_positions.extend(block)

        sampled_positions = sampled_positions[:n_test]
        sampled_idx = test_idx[sampled_positions]

        # fit model on full training set (everything before test)
        # assume test_idx is tail segment, so train is before it
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

        model_sent = make_logreg(C=model_C)
        model_ns = make_logreg(C=model_C)

        model_sent.fit(X_sent.iloc[train_idx], y.iloc[train_idx])
        model_ns.fit(X_nosent.iloc[train_idx], y.iloc[train_idx])

        pred_sent = model_sent.predict(X_sent.iloc[sampled_idx])
        pred_ns = model_ns.predict(X_nosent.iloc[sampled_idx])

        f1_sent = f1_score(y.iloc[sampled_idx], pred_sent, average="macro")
        f1_ns = f1_score(y.iloc[sampled_idx], pred_ns, average="macro")

        deltas.append(f1_sent - f1_ns)

    deltas = np.array(deltas)

    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)

    p_like = float(np.mean(deltas <= 0))  # probability delta is not positive

    return {
        "bootstrap_mean_delta": float(np.mean(deltas)),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_p_like": p_like
    }


# -----------------------------
# (C) Likelihood Ratio Test
# -----------------------------
def likelihood_ratio_test(X_sent, X_nosent, y, test_idx, C=1.0):
    """
    Likelihood ratio test using log-likelihood on test set.

    H0: NoSent is sufficient
    H1: Sent improves likelihood

    LR = 2*(LL_sent - LL_nosent)
    df = (#features_sent - #features_nosent)
    p = 1 - chi2.cdf(LR, df)
    """

    test_idx = np.array(test_idx)
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    model_sent = make_logreg(C=C)
    model_ns = make_logreg(C=C)

    model_sent.fit(X_sent.iloc[train_idx], y.iloc[train_idx])
    model_ns.fit(X_nosent.iloc[train_idx], y.iloc[train_idx])

    proba_sent = model_sent.predict_proba(X_sent.iloc[test_idx])
    proba_ns = model_ns.predict_proba(X_nosent.iloc[test_idx])

    # log_loss returns negative log-likelihood average
    nll_sent = log_loss(y.iloc[test_idx], proba_sent, labels=np.unique(y))
    nll_ns = log_loss(y.iloc[test_idx], proba_ns, labels=np.unique(y))

    # convert to total LL
    LL_sent = -nll_sent * len(test_idx)
    LL_ns = -nll_ns * len(test_idx)

    LR = 2.0 * (LL_sent - LL_ns)

    df_diff = X_sent.shape[1] - X_nosent.shape[1]
    if df_diff <= 0:
        raise ValueError("X_sent must have more features than X_nosent for LR test.")

    p_value = 1 - chi2.cdf(LR, df=df_diff)

    return {
        "LL_sent": float(LL_sent),
        "LL_nosent": float(LL_ns),
        "LR_stat": float(LR),
        "df_diff": int(df_diff),
        "p_value": float(p_value)
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet("/kaggle/input/sec-filings-with-stock-price-features/us_market_dataset.parquet")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    print("Dataset shape:", df.shape)

    X_sent = build_feature_matrix(df, drop_sentiment=False)
    X_nosent = build_feature_matrix(df, drop_sentiment=True)

    print("Features WITH sentiment:", X_sent.shape)
    print("Features WITHOUT sentiment:", X_nosent.shape)

    # fixed test tail
    test_ratio = 0.2
    test_size = int(len(df) * test_ratio)
    test_idx = np.arange(len(df) - test_size, len(df))

    # CV splitter on train/val region only
    train_val_idx = np.arange(0, len(df) - test_size)
    tscv = TimeSeriesSplit(n_splits=20)

    print("\n================ SIGNIFICANCE TESTS (LOGREG) ================\n")

    for horizon in range(1, 8):
        label_col = f"label_{horizon}d"
        if label_col not in df.columns:
            continue

        y = df[label_col].astype(int)

        # build CV views (train+val only)
        X_sent_tv = X_sent.iloc[train_val_idx].reset_index(drop=True)
        X_ns_tv = X_nosent.iloc[train_val_idx].reset_index(drop=True)
        y_tv = y.iloc[train_val_idx].reset_index(drop=True)

        print(f"\n================ Horizon {horizon} Day(s) ================\n")

        # (A) Paired fold test
        paired_res = paired_fold_test(X_sent_tv, X_ns_tv, y_tv, tscv, C=1.0)

        print("---- Paired Fold Test (CV) ----")
        print("Fold deltas:", np.round(paired_res["delta_fold_f1"], 6))
        print("Mean ΔF1:", paired_res["mean_delta"])
        print("Paired t-test p-value:", paired_res["ttest_p"])
        print("Wilcoxon p-value:", paired_res["wilcoxon_p"])

        # (B) Block bootstrap on test tail
        boot_res = block_bootstrap_delta_f1(
            X_sent, X_nosent, y,
            test_idx=test_idx,
            model_C=1.0,
            n_boot=300,
            block_size=200
        )

        print("\n---- Block Bootstrap Test (Test Tail) ----")
        print("Bootstrap mean ΔF1:", boot_res["bootstrap_mean_delta"])
        print("95% CI:", (boot_res["bootstrap_ci_low"], boot_res["bootstrap_ci_high"]))
        print("p-like (fraction Δ<=0):", boot_res["bootstrap_p_like"])

        # (C) Likelihood Ratio test
        lr_res = likelihood_ratio_test(X_sent, X_nosent, y, test_idx=test_idx, C=1.0)

        print("\n---- Likelihood Ratio Test (Test Tail) ----")
        print("LL Sent:", lr_res["LL_sent"])
        print("LL NoSent:", lr_res["LL_nosent"])
        print("LR stat:", lr_res["LR_stat"])
        print("df diff:", lr_res["df_diff"])
        print("p-value:", lr_res["p_value"])