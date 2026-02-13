import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from tabulate import tabulate

# XGBoost
from xgboost import XGBClassifier, XGBRegressor


# -----------------------------
# Expanding window split
# -----------------------------
class ExpandingWindowSplitter:
    def __init__(self, n_splits=5, test_ratio=0.2):
        self.n_splits = n_splits
        self.test_ratio = test_ratio

    def split(self, df, date_col):
        df = df.sort_values(date_col).reset_index(drop=True)

        n = len(df)
        test_size = int(n * self.test_ratio)

        train_val = df.iloc[:-test_size]
        test = df.iloc[-test_size:]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        return train_val, test, tscv


# -----------------------------
# Generic trainer
# -----------------------------
class ModelTrainer:
    def __init__(self, model, task="classification", scale=False):
        self.task = task
        self.scale = scale

        if scale:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            self.pipeline = Pipeline([
                ("model", model)
            ])

    def cross_validate(self, X, y, splitter):
        scores = []

        for train_idx, val_idx in splitter.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.pipeline.fit(X_train, y_train)
            preds = self.pipeline.predict(X_val)

            if self.task == "classification":
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average="macro")
                scores.append((acc, f1))
            else:
                mae = mean_absolute_error(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                r2 = r2_score(y_val, preds)
                scores.append((mae, rmse, r2))

        return np.mean(scores, axis=0)

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def evaluate(self, X, y):
        preds = self.pipeline.predict(X)

        if self.task == "classification":
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="macro")
            return acc, f1
        else:
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            return mae, rmse, r2


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_parquet("us_market_dataset.parquet")

DATE_COL = "filing_date"
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

print("Dataset shape:", df.shape)


# -----------------------------
# Define columns
# -----------------------------
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


X_with_sentiment = build_feature_matrix(df, drop_sentiment=False)
X_without_sentiment = build_feature_matrix(df, drop_sentiment=True)

print("Features WITH sentiment:", X_with_sentiment.shape)
print("Features WITHOUT sentiment:", X_without_sentiment.shape)


# -----------------------------
# Split dataset
# -----------------------------
splitter = ExpandingWindowSplitter(n_splits=5, test_ratio=0.2)
train_val_df, test_df, tscv = splitter.split(df, DATE_COL)

X_train_val_with = X_with_sentiment.loc[train_val_df.index]
X_test_with = X_with_sentiment.loc[test_df.index]

X_train_val_wo = X_without_sentiment.loc[train_val_df.index]
X_test_wo = X_without_sentiment.loc[test_df.index]


# ============================================================
# MODELS
# ============================================================
classification_models = {
    "LogReg": (LogisticRegression(max_iter=3000), True),
    "RandomForest": (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), False),
    "XGBoost": (
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        False
    )
}

regression_models = {
    "Ridge": (Ridge(alpha=1.0), True),
    "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), False),
    "XGBoost": (
        XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        False
    )
}


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================
print("\n================ MULTICLASS CLASSIFICATION RESULTS ================\n")

for day in range(1, 8):
    label_col = f"label_{day}d"
    if label_col not in df.columns:
        continue

    y_train_val = df.loc[train_val_df.index, label_col].astype(int)
    y_test = df.loc[test_df.index, label_col].astype(int)

    rows = []

    for model_name, (model, use_scaler) in classification_models.items():
        # WITH sentiment
        trainer_with = ModelTrainer(model=model, task="classification", scale=use_scaler)
        cv_acc_w, cv_f1_w = trainer_with.cross_validate(X_train_val_with, y_train_val, tscv)
        trainer_with.fit(X_train_val_with, y_train_val)
        test_acc_w, test_f1_w = trainer_with.evaluate(X_test_with, y_test)

        # WITHOUT sentiment
        trainer_wo = ModelTrainer(model=model, task="classification", scale=use_scaler)
        cv_acc_wo, cv_f1_wo = trainer_wo.cross_validate(X_train_val_wo, y_train_val, tscv)
        trainer_wo.fit(X_train_val_wo, y_train_val)
        test_acc_wo, test_f1_wo = trainer_wo.evaluate(X_test_wo, y_test)

        rows.append([
            model_name,
            f"{cv_acc_w:.4f}", f"{cv_f1_w:.4f}",
            f"{test_acc_w:.4f}", f"{test_f1_w:.4f}",
            f"{test_acc_wo:.4f}", f"{test_f1_wo:.4f}",
            f"{(test_f1_w - test_f1_wo):+.4f}"
        ])

    print(f"\n--- Horizon {day} Day(s) ---\n")
    print(tabulate(
        rows,
        headers=[
            "Model",
            "CV Acc (Sent)", "CV F1 (Sent)",
            "Test Acc (Sent)", "Test F1 (Sent)",
            "Test Acc (NoSent)", "Test F1 (NoSent)",
            "Δ Test F1"
        ],
        tablefmt="github"
    ))


# ============================================================
# REGRESSION RESULTS
# ============================================================
print("\n================ REGRESSION RESULTS ================\n")

for day in range(1, 8):
    target_col = f"return_{day}d"
    if target_col not in df.columns:
        continue

    y_train_val = df.loc[train_val_df.index, target_col].astype(float)
    y_test = df.loc[test_df.index, target_col].astype(float)

    rows = []

    for model_name, (model, use_scaler) in regression_models.items():
        # WITH sentiment
        trainer_with = ModelTrainer(model=model, task="regression", scale=use_scaler)
        cv_mae_w, cv_rmse_w, cv_r2_w = trainer_with.cross_validate(X_train_val_with, y_train_val, tscv)
        trainer_with.fit(X_train_val_with, y_train_val)
        test_mae_w, test_rmse_w, test_r2_w = trainer_with.evaluate(X_test_with, y_test)

        # WITHOUT sentiment
        trainer_wo = ModelTrainer(model=model, task="regression", scale=use_scaler)
        cv_mae_wo, cv_rmse_wo, cv_r2_wo = trainer_wo.cross_validate(X_train_val_wo, y_train_val, tscv)
        trainer_wo.fit(X_train_val_wo, y_train_val)
        test_mae_wo, test_rmse_wo, test_r2_wo = trainer_wo.evaluate(X_test_wo, y_test)

        rows.append([
            model_name,
            f"{test_mae_w:.6f}", f"{test_rmse_w:.6f}", f"{test_r2_w:.4f}",
            f"{test_mae_wo:.6f}", f"{test_rmse_wo:.6f}", f"{test_r2_wo:.4f}",
            f"{(test_rmse_w - test_rmse_wo):+.6f}"
        ])

    print(f"\n--- Horizon {day} Day(s) ---\n")
    print(tabulate(
        rows,
        headers=[
            "Model",
            "Test MAE (Sent)", "Test RMSE (Sent)", "Test R2 (Sent)",
            "Test MAE (NoSent)", "Test RMSE (NoSent)", "Test R2 (NoSent)",
            "Δ Test RMSE"
        ],
        tablefmt="github"
    ))