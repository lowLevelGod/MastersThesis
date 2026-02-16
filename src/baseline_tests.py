import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def most_frequent_class_baseline(df, train_idx, test_idx, label_cols):
    results = []

    for col in label_cols:
        y_train = df.loc[train_idx, col].astype(int)
        y_test = df.loc[test_idx, col].astype(int)

        most_common_label = y_train.value_counts().idxmax()

        preds = np.full(len(y_test), most_common_label)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        results.append({
            "label_col": col,
            "most_common_label": int(most_common_label),
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1)
        })

    return pd.DataFrame(results)


# Example usage
df = pd.read_parquet("us_market_dataset.parquet")
df["filing_date"] = pd.to_datetime(df["filing_date"])
df = df.sort_values("filing_date").reset_index(drop=True)

label_cols = [f"label_{d}d" for d in range(1, 8)]

# Expanding style split: last 20% test
test_ratio = 0.2
n = len(df)
test_size = int(n * test_ratio)

train_idx = df.index[:-test_size]
test_idx = df.index[-test_size:]

baseline_df = most_frequent_class_baseline(df, train_idx, test_idx, label_cols)

print(baseline_df)
baseline_df.to_csv("baseline_most_frequent_class.csv", index=False)