import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Define columns
# ============================================================

NOT_NEEDED_COLS = [
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
    "event_label",
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
    "event_label",
    "sector",
    "industry"
]

SENTIMENT_COLS = [
    "finbert_neg",
    "finbert_neu",
    "finbert_pos",
    "finbert_neg_std",
    "finbert_neu_std",
    "finbert_pos_std",
    "finbert_polarity_std"
]

CLASSIFICATION_LABEL_COLS = [f"label_{day}d" for day in range(1, 8)]
REGRESSION_LABEL_COLS = [f"return_{day}d" for day in range(1, 8)]

DROP_COLS = NON_FEATURE_COLS + CLASSIFICATION_LABEL_COLS + REGRESSION_LABEL_COLS

# ============================================================
# Utilities
# ============================================================

def reset_csv(csv_path):
    if os.path.exists(csv_path):
        os.remove(csv_path)

def append_row_to_csv(csv_path, row_dict):
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)

# ============================================================
# Models
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, task="classification"):
        super().__init__()
        self.task = task

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        if task == "classification":
            self.fc = nn.Linear(hidden_dim, 3)
        else:
            self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ---------------------------
# PatchTST (simplified)
# ---------------------------

class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, patch_size=4, d_model=64, task="classification"):
        super().__init__()
        self.task = task
        self.patch_size = patch_size

        self.num_patches = seq_len // patch_size

        self.proj = nn.Linear(input_dim * patch_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if task == "classification":
            self.fc = nn.Linear(d_model, 3)
        else:
            self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, F = x.shape

        # trim to divisible length
        T_new = (T // self.patch_size) * self.patch_size
        x = x[:, :T_new, :]

        x = x.reshape(B, -1, self.patch_size * F)
        x = self.proj(x)

        x = self.transformer(x)

        x = x.mean(dim=1)  # pooling
        return self.fc(x)

# ============================================================
# Training
# ============================================================

def train_model(model, X_train, y_train, X_test, y_test, task):

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train).to(DEVICE)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test).to(DEVICE)

    model.to(DEVICE)
    model.train()

    for _ in range(10):
        optimizer.zero_grad()

        outputs = model(X_train)

        if task == "classification":
            loss = criterion(outputs, y_train.long())
        else:
            loss = criterion(outputs.squeeze(), y_train.float())

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)

        if task == "classification":
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            return f1_score(y_test.cpu(), preds, average="macro")

        else:
            preds = preds.squeeze().cpu().numpy()
            return np.sqrt(mean_squared_error(y_test.cpu(), preds))

# ============================================================
# Sequence Builder
# ============================================================

def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

# ============================================================
# Feature Builder
# ============================================================

def build_features(df, drop_cols, drop_sentiment=False, sentiment_cols=None):
    X = df.drop(columns=drop_cols, errors="ignore")

    if drop_sentiment:
        X = X.drop(columns=sentiment_cols, errors="ignore")

    return X.select_dtypes(include=[np.number]).fillna(0.0)

# ============================================================
# Split
# ============================================================

def split_sequences(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]

# ============================================================
# Main Experiment
# ============================================================

# ============================================================
# Aggregation for sector / industry time-series
# ============================================================

def aggregate_group_timeseries(df_group, group_col, task):
    """
    Aggregates multiple filings per day into one time step.
    Works for sector / industry level modeling.
    """

    df_group = df_group.copy()
    df_group["filing_date"] = pd.to_datetime(df_group["filing_date"])

    numeric_cols = df_group.select_dtypes(include=[np.number]).columns.tolist()

    # remove labels from feature aggregation
    numeric_cols = [
        c for c in numeric_cols
        if not c.startswith("label_") and not c.startswith("return_")
    ]

    # aggregate features (mean)
    df_feat = (
        df_group
        .groupby("filing_date")[numeric_cols]
        .mean()
    )

    # aggregate labels
    label_cols = (
        CLASSIFICATION_LABEL_COLS
        if task == "classification"
        else REGRESSION_LABEL_COLS
    )

    df_labels = []

    for col in label_cols:

        if task == "classification":
            agg = (
                df_group
                .groupby("filing_date")[col]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            )
        else:
            agg = (
                df_group
                .groupby("filing_date")[col]
                .mean()
            )

        df_labels.append(agg)

    df_labels = pd.concat(df_labels, axis=1)

    df_out = pd.concat([df_feat, df_labels], axis=1).reset_index()

    return df_out.sort_values("filing_date")

def run_experiment(df, group_col, task="classification"):

    SEQ_LENGTHS = [5, 10, 15, 20]
    MODELS = ["LSTM", "PatchTST"]

    groups = df[group_col].dropna().unique()

    for seq_len in SEQ_LENGTHS:
        for model_name in MODELS:

            csv_path = f"results_{group_col}_{task}_{model_name}_seq{seq_len}.csv"
            reset_csv(csv_path)

            for group in groups:

                df_group = df[df[group_col] == group].copy()

                if len(df_group) < 60:
                    continue

                # -----------------------------------------
                # IMPORTANT: aggregate for sector/industry
                # -----------------------------------------
                if group_col in ["sector", "industry"]:
                    df_group = aggregate_group_timeseries(df_group, group_col, task)

                df_group = df_group.sort_values("filing_date").reset_index(drop=True)

                X_with = build_features(df_group, DROP_COLS, False, SENTIMENT_COLS)
                X_no = build_features(df_group, DROP_COLS, True, SENTIMENT_COLS)

                scaler_w = StandardScaler()
                scaler_n = StandardScaler()

                X_with = scaler_w.fit_transform(X_with)
                X_no = scaler_n.fit_transform(X_no)

                for day in range(1, 8):

                    label_col = f"label_{day}d" if task == "classification" else f"return_{day}d"

                    if label_col not in df_group.columns:
                        continue

                    y = df_group[label_col].values

                    X_seq_w, y_seq = build_sequences(X_with, y, seq_len)
                    X_seq_n, _ = build_sequences(X_no, y, seq_len)

                    if len(X_seq_w) < 50:
                        continue

                    X_train_w, X_test_w, y_train, y_test = split_sequences(X_seq_w, y_seq)
                    X_train_n, X_test_n, _, _ = split_sequences(X_seq_n, y_seq)

                    # -------------------
                    # Model selection
                    # -------------------
                    if model_name == "LSTM":
                        model_w = LSTMModel(X_train_w.shape[2], task=task)
                        model_n = LSTMModel(X_train_n.shape[2], task=task)

                    else:
                        model_w = PatchTST(X_train_w.shape[2], seq_len, task=task)
                        model_n = PatchTST(X_train_n.shape[2], seq_len, task=task)

                    score_w = train_model(model_w, X_train_w, y_train, X_test_w, y_test, task)
                    score_n = train_model(model_n, X_train_n, y_train, X_test_n, y_test, task)

                    delta = score_w - score_n

                    # -------------------
                    # Baseline
                    # -------------------
                    if task == "classification":
                        baseline_class = pd.Series(y_train).mode()[0]
                        baseline_preds = np.full_like(y_test, baseline_class)
                        baseline_score = f1_score(y_test, baseline_preds, average="macro")
                    else:
                        baseline_val = y_train.mean()
                        baseline_preds = np.full_like(y_test, baseline_val)
                        baseline_score = np.sqrt(mean_squared_error(y_test, baseline_preds))

                    row = {
                        group_col: group,
                        "horizon_days": day,
                        "seq_len": seq_len,
                        "samples": len(df_group),
                        "model": model_name,
                        "test_no_sent": score_n,
                        "test_with_sent": score_w,
                        "delta": delta,
                        "baseline": baseline_score
                    }

                    append_row_to_csv(csv_path, row)

            print(f"Finished {model_name} | seq_len={seq_len} | {task}")

# ============================================================
# RUN
# ============================================================

df = pd.read_parquet("us_market_dataset_with_sector.parquet")

df.drop(columns=NOT_NEEDED_COLS, inplace=True)

df["filing_date"] = pd.to_datetime(df["filing_date"])
df = df.sort_values("filing_date")

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, wilcoxon

def set_publication_style():
    sns.set_theme(style="white")  # clean white background
    
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
    })

def analyze_all_lstm_experiments(group_col):

    set_publication_style()

    tasks = ["classification", "regression"]
    models = ["LSTM", "PatchTST"]
    
    summary_results = []

    # -------------------------------------------------------
    # Run analysis per experiment
    # -------------------------------------------------------

    for task in tasks:
        for model_name in models:
            for seq_len in [5, 10, 15, 20]:
                
                csv_path = f"results_{group_col}_{task}_{model_name}_seq{seq_len}.csv"

                if not os.path.exists(csv_path):
                    print(f"CSV not found: {csv_path}")
                    continue

                df_res = pd.read_csv(csv_path)

                print("\n" + "="*80)
                print(f"{group_col.upper()} | {task.upper()} | {model_name} | SEQ={seq_len}")
                print("="*80)

                # -------------------------------------------------------
                # 2️⃣ PER-HORIZON ANALYSIS
                # -------------------------------------------------------

                print("\n--- Per-Horizon Inference ---")

                for h in sorted(df_res["horizon_days"].unique()):

                    subset = df_res[df_res["horizon_days"] == h]
                    deltas = subset["delta"].dropna()

                    if len(deltas) < 5:
                        continue

                    mean_delta = deltas.mean()
                    std_delta = deltas.std()
                    n = len(deltas)

                    ci_low = mean_delta - 1.96 * std_delta / np.sqrt(n)
                    ci_high = mean_delta + 1.96 * std_delta / np.sqrt(n)

                    t_stat, p_val = ttest_1samp(deltas, 0)

                    if len(deltas) > 10:
                        _, w_p = wilcoxon(deltas)
                    else:
                        w_p = np.nan

                    print(f"\nHorizon {h} days")
                    print(f"N groups: {n}")
                    print(f"Mean Δ: {mean_delta:.6f}")
                    print(f"95% CI: ({ci_low:.6f}, {ci_high:.6f})")
                    print(f"T-test p-value: {p_val:.6f}")
                    print(f"Wilcoxon p-value: {w_p}")

                # -------------------------------------------------------
                # 3️⃣ OVERALL EFFECT (correct aggregation)
                # -------------------------------------------------------

                print("\n--- Overall Effect (Averaged Across Horizons Per Group) ---")

                avg_per_group = (
                    df_res.groupby(group_col)["delta"]
                    .mean()
                    .dropna()
                )

                deltas = avg_per_group.values

                if len(deltas) >= 5:

                    mean_delta = np.mean(deltas)
                    std_delta = np.std(deltas)
                    n = len(deltas)

                    ci_low = mean_delta - 1.96 * std_delta / np.sqrt(n)
                    ci_high = mean_delta + 1.96 * std_delta / np.sqrt(n)

                    t_stat, p_val = ttest_1samp(deltas, 0)

                    if len(deltas) > 10:
                        _, w_p = wilcoxon(deltas)
                    else:
                        w_p = np.nan

                    print(f"N groups: {n}")
                    print(f"Mean Δ (avg across horizons): {mean_delta:.6f}")
                    print(f"95% CI: ({ci_low:.6f}, {ci_high:.6f})")
                    print(f"T-test p-value: {p_val:.6f}")
                    print(f"Wilcoxon p-value: {w_p}")

                # -------------------------------------------------------
                # 4️⃣ TOP / BOTTOM GROUPS
                # -------------------------------------------------------

                print("\n--- Top & Bottom Performing Groups ---")

                ranked = avg_per_group.sort_values(ascending=False)

                print("\nTop 5 groups:")
                print(ranked.head(5))

                print("\nBottom 5 groups:")
                print(ranked.tail(5))

                # -------------------------------------------------------
                # 5️⃣ BASELINE COMPARISON
                # -------------------------------------------------------

                if task == "classification":
                    beats = (df_res["test_with_sent"] > df_res["baseline"]).mean()
                else:
                    beats = (df_res["test_with_sent"] < df_res["baseline"]).mean()

                print("\nBeats baseline percentage:", round(beats * 100, 2), "%")

                # -------------------------------------------------------
                # 6️⃣ SEQ LENGTH EFFECT (optional insight)
                # -------------------------------------------------------

                print("\n--- Summary ---")
                print(f"Model: {model_name} | Sequence Length: {seq_len}")
                print(f"Mean Δ overall: {df_res['delta'].mean():.6f}")
                
                summary_results.append({
                    "group": group_col,
                    "task": task,
                    "model": model_name,
                    "seq_len": seq_len,
                    "mean_delta": avg_per_group.mean()
                })
                
    # =======================================================
    # FINAL COMPARISON PLOT (LSTM vs PatchTST vs SEQ LEN)
    # =======================================================

    summary_df = pd.DataFrame(summary_results)

    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------------
    # Classification plot
    # ---------------------------
    ax = axes[0]

    df_cls = summary_df[summary_df["task"] == "classification"]

    for model_name in ["LSTM", "PatchTST"]:
        subset = df_cls[df_cls["model"] == model_name]
        subset = subset.sort_values("seq_len")

        ax.plot(
            subset["seq_len"],
            subset["mean_delta"],
            marker="o",
            label=model_name
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Classification: Mean Δ vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean Overall Δ")
    ax.legend()

    # ---------------------------
    # Regression plot
    # ---------------------------
    ax = axes[1]

    df_reg = summary_df[summary_df["task"] == "regression"]

    for model_name in ["LSTM", "PatchTST"]:
        subset = df_reg[df_reg["model"] == model_name]
        subset = subset.sort_values("seq_len")

        ax.plot(
            subset["seq_len"],
            subset["mean_delta"],
            marker="o",
            label=model_name
        )
        
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Regression: Mean Δ vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean Overall Δ")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{group_col}_sequence_length_model_comparison.pdf")
    plt.close()

# Run experiments
# ticker level
# run_experiment(df, "ticker", task="classification")
# run_experiment(df, "ticker", task="regression")

# sector level
# run_experiment(df, "sector", task="classification")
# run_experiment(df, "sector", task="regression")

# industry level
# run_experiment(df, "industry", task="classification")
# run_experiment(df, "industry", task="regression")

analyze_all_lstm_experiments("ticker")
analyze_all_lstm_experiments("sector")
analyze_all_lstm_experiments("industry")