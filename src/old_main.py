import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from xgboost import XGBClassifier, XGBRegressor

from tabulate import tabulate


# ============================================================
# Splitter
# ============================================================
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


# ============================================================
# Trainer
# ============================================================
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


# ============================================================
# Grid Search Base Class
# ============================================================
class BaseGridSearch:
    def __init__(self, task):
        self.task = task

    def get_param_grid(self):
        raise NotImplementedError

    def build_model(self, params):
        raise NotImplementedError

    def scale_required(self):
        return False

    def score(self, cv_metrics):
        """
        Choose what "best" means.
        Classification: maximize F1
        Regression: minimize RMSE
        """
        if self.task == "classification":
            cv_acc, cv_f1 = cv_metrics
            return cv_f1
        else:
            cv_mae, cv_rmse, cv_r2 = cv_metrics
            return -cv_rmse  # negative so bigger = better

    def run(self, X_train_val, y_train_val, splitter):
        best_params = None
        best_cv_metrics = None
        best_score = -np.inf

        for params in ParameterGrid(self.get_param_grid()):
            model = self.build_model(params)
            trainer = ModelTrainer(
                model=model,
                task=self.task,
                scale=self.scale_required()
            )

            cv_metrics = trainer.cross_validate(X_train_val, y_train_val, splitter)
            s = self.score(cv_metrics)

            if s > best_score:
                best_score = s
                best_params = params
                best_cv_metrics = cv_metrics

        return best_params, best_cv_metrics


# ============================================================
# Logistic Regression Grid Search
# ============================================================
class LogisticRegressionGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="classification")

    def scale_required(self):
        return True

    def get_param_grid(self):
        return {
            "C": [0.01, 0.1, 1.0, 5.0],
            "solver": ["lbfgs"],
            "max_iter": [4000]
        }

    def build_model(self, params):
        return LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            max_iter=params["max_iter"],
            multi_class="multinomial"
        )


# ============================================================
# Random Forest Grid Search (Classification)
# ============================================================
class RandomForestClassifierGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="classification")

    def get_param_grid(self):
        return {
            "n_estimators": [200, 500, 1000, 2000],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3]
        }

    def build_model(self, params):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )


# ============================================================
# XGBoost Grid Search (Classification)
# ============================================================
class XGBoostClassifierGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="classification")

    def get_param_grid(self):
        return {
            "n_estimators": [200, 500, 1000, 2000],
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

    def build_model(self, params):
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )


# ============================================================
# Ridge Grid Search (Regression)
# ============================================================
class RidgeGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="regression")

    def scale_required(self):
        return True

    def get_param_grid(self):
        return {
            "alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
        }

    def build_model(self, params):
        return Ridge(alpha=params["alpha"])


# ============================================================
# Random Forest Grid Search (Regression)
# ============================================================
class RandomForestRegressorGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="regression")

    def get_param_grid(self):
        return {
            "n_estimators": [200, 500, 1000, 2000],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 3]
        }

    def build_model(self, params):
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )


# ============================================================
# XGBoost Grid Search (Regression)
# ============================================================
class XGBoostRegressorGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="regression")

    def get_param_grid(self):
        return {
            "n_estimators": [200, 500, 1000, 2000],
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

    def build_model(self, params):
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )


# ============================================================
# Utilities
# ============================================================
def improvement_tag(delta):
    return "improved" if delta > 0 else "worse"


def append_row_to_csv(csv_path, row_dict):
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)


def build_feature_matrix(df, drop_cols, drop_sentiment=False, sentiment_cols=None):
    X = df.drop(columns=drop_cols, errors="ignore")

    if drop_sentiment and sentiment_cols is not None:
        X = X.drop(columns=sentiment_cols, errors="ignore")

    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X


# ============================================================
# Load dataset
# ============================================================
df = pd.read_parquet("/kaggle/input/sec-filings-with-stock-price-features/us_market_dataset.parquet")

DATE_COL = "filing_date"
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

print("Dataset shape:", df.shape)


# ============================================================
# Define columns
# ============================================================
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

DROP_COLS = NON_FEATURE_COLS + CLASSIFICATION_LABEL_COLS + REGRESSION_LABEL_COLS


# ============================================================
# Build feature sets
# ============================================================
X_with_sent = build_feature_matrix(df, DROP_COLS, drop_sentiment=False, sentiment_cols=SENTIMENT_COLS)
X_no_sent = build_feature_matrix(df, DROP_COLS, drop_sentiment=True, sentiment_cols=SENTIMENT_COLS)

print("Features WITH sentiment:", X_with_sent.shape)
print("Features WITHOUT sentiment:", X_no_sent.shape)


# ============================================================
# Split
# ============================================================
splitter = ExpandingWindowSplitter(n_splits=5, test_ratio=0.2)
train_val_df, test_df, tscv = splitter.split(df, DATE_COL)

X_train_val_with = X_with_sent.loc[train_val_df.index]
X_test_with = X_with_sent.loc[test_df.index]

X_train_val_no = X_no_sent.loc[train_val_df.index]
X_test_no = X_no_sent.loc[test_df.index]

print("\nTrain/Val size:", len(train_val_df))
print("Test size:", len(test_df))


# ============================================================
# Grid Search Registries
# ============================================================
classification_searchers = {
    "LogReg": LogisticRegressionGridSearch(),
    "RandomForest": RandomForestClassifierGridSearch(),
    "XGBoost": XGBoostClassifierGridSearch()
}

regression_searchers = {
    "Ridge": RidgeGridSearch(),
    "RandomForest": RandomForestRegressorGridSearch(),
    "XGBoost": XGBoostRegressorGridSearch()
}


# ============================================================
# CLASSIFICATION GRID SEARCH
# ============================================================
print("\n================ CLASSIFICATION GRID SEARCH RESULTS ================\n")

for day in range(1, 8):
    label_col = f"label_{day}d"
    if label_col not in df.columns:
        continue

    y_train_val = df.loc[train_val_df.index, label_col].astype(int)
    y_test = df.loc[test_df.index, label_col].astype(int)

    rows = []

    for model_name, searcher in classification_searchers.items():
        print(f"\n[Classification] Horizon={day}d Model={model_name} | Grid search NO SENTIMENT...")
        best_params_no, best_cv_no = searcher.run(X_train_val_no, y_train_val, tscv)

        best_model_no = searcher.build_model(best_params_no)
        trainer_no = ModelTrainer(best_model_no, task="classification", scale=searcher.scale_required())
        trainer_no.fit(X_train_val_no, y_train_val)
        test_acc_no, test_f1_no = trainer_no.evaluate(X_test_no, y_test)

        print(f"[Classification] Horizon={day}d Model={model_name} | Grid search WITH SENTIMENT...")
        best_params_sent, best_cv_sent = searcher.run(X_train_val_with, y_train_val, tscv)

        best_model_sent = searcher.build_model(best_params_sent)
        trainer_sent = ModelTrainer(best_model_sent, task="classification", scale=searcher.scale_required())
        trainer_sent.fit(X_train_val_with, y_train_val)
        test_acc_sent, test_f1_sent = trainer_sent.evaluate(X_test_with, y_test)

        cv_acc_no, cv_f1_no = best_cv_no
        cv_acc_sent, cv_f1_sent = best_cv_sent

        delta_cv_f1 = cv_f1_sent - cv_f1_no
        delta_test_f1 = test_f1_sent - test_f1_no

        rows.append([
            model_name,
            f"{cv_acc_no:.4f}", f"{cv_f1_no:.4f}",
            f"{cv_acc_sent:.4f}", f"{cv_f1_sent:.4f}",
            f"{test_acc_no:.4f}", f"{test_f1_no:.4f}",
            f"{test_acc_sent:.4f}", f"{test_f1_sent:.4f}",
            f"{delta_test_f1:+.4f} ({improvement_tag(delta_test_f1)})"
        ])

        out_csv = f"classification_gridsearch_{model_name}_horizon_{day}d.csv"

        row_dict = {
            "task": "classification",
            "horizon_days": day,
            "model": model_name,

            "best_params_no_sent": json.dumps(best_params_no),
            "best_params_with_sent": json.dumps(best_params_sent),

            "cv_acc_no_sent": float(cv_acc_no),
            "cv_f1_no_sent": float(cv_f1_no),
            "cv_acc_with_sent": float(cv_acc_sent),
            "cv_f1_with_sent": float(cv_f1_sent),

            "test_acc_no_sent": float(test_acc_no),
            "test_f1_no_sent": float(test_f1_no),
            "test_acc_with_sent": float(test_acc_sent),
            "test_f1_with_sent": float(test_f1_sent),

            "delta_cv_f1": float(delta_cv_f1),
            "delta_test_f1": float(delta_test_f1)
        }

        append_row_to_csv(out_csv, row_dict)
        print(f"Saved row -> {out_csv}")

    print(f"\n--- Classification Horizon {day} Day(s) ---\n")
    print(tabulate(
        rows,
        headers=[
            "Model",
            "CV Acc (NoSent)", "CV F1 (NoSent)",
            "CV Acc (Sent)", "CV F1 (Sent)",
            "Test Acc (NoSent)", "Test F1 (NoSent)",
            "Test Acc (Sent)", "Test F1 (Sent)",
            "Δ Test F1"
        ],
        tablefmt="github"
    ))


# ============================================================
# REGRESSION GRID SEARCH
# ============================================================
print("\n================ REGRESSION GRID SEARCH RESULTS ================\n")

for day in range(1, 8):
    target_col = f"return_{day}d"
    if target_col not in df.columns:
        continue

    y_train_val = df.loc[train_val_df.index, target_col].astype(float)
    y_test = df.loc[test_df.index, target_col].astype(float)

    rows = []

    for model_name, searcher in regression_searchers.items():
        print(f"\n[Regression] Horizon={day}d Model={model_name} | Grid search NO SENTIMENT...")
        best_params_no, best_cv_no = searcher.run(X_train_val_no, y_train_val, tscv)

        best_model_no = searcher.build_model(best_params_no)
        trainer_no = ModelTrainer(best_model_no, task="regression", scale=searcher.scale_required())
        trainer_no.fit(X_train_val_no, y_train_val)
        test_mae_no, test_rmse_no, test_r2_no = trainer_no.evaluate(X_test_no, y_test)

        print(f"[Regression] Horizon={day}d Model={model_name} | Grid search WITH SENTIMENT...")
        best_params_sent, best_cv_sent = searcher.run(X_train_val_with, y_train_val, tscv)

        best_model_sent = searcher.build_model(best_params_sent)
        trainer_sent = ModelTrainer(best_model_sent, task="regression", scale=searcher.scale_required())
        trainer_sent.fit(X_train_val_with, y_train_val)
        test_mae_sent, test_rmse_sent, test_r2_sent = trainer_sent.evaluate(X_test_with, y_test)

        cv_mae_no, cv_rmse_no, cv_r2_no = best_cv_no
        cv_mae_sent, cv_rmse_sent, cv_r2_sent = best_cv_sent

        delta_cv_rmse = cv_rmse_sent - cv_rmse_no
        delta_test_rmse = test_rmse_sent - test_rmse_no

        rows.append([
            model_name,
            f"{cv_rmse_no:.6f}",
            f"{cv_rmse_sent:.6f}",
            f"{test_rmse_no:.6f}",
            f"{test_rmse_sent:.6f}",
            f"{delta_test_rmse:+.6f} ({improvement_tag(-delta_test_rmse)})"
        ])

        out_csv = f"regression_gridsearch_{model_name}_horizon_{day}d.csv"

        row_dict = {
            "task": "regression",
            "horizon_days": day,
            "model": model_name,

            "best_params_no_sent": json.dumps(best_params_no),
            "best_params_with_sent": json.dumps(best_params_sent),

            "cv_mae_no_sent": float(cv_mae_no),
            "cv_rmse_no_sent": float(cv_rmse_no),
            "cv_r2_no_sent": float(cv_r2_no),

            "cv_mae_with_sent": float(cv_mae_sent),
            "cv_rmse_with_sent": float(cv_rmse_sent),
            "cv_r2_with_sent": float(cv_r2_sent),

            "test_mae_no_sent": float(test_mae_no),
            "test_rmse_no_sent": float(test_rmse_no),
            "test_r2_no_sent": float(test_r2_no),

            "test_mae_with_sent": float(test_mae_sent),
            "test_rmse_with_sent": float(test_rmse_sent),
            "test_r2_with_sent": float(test_r2_sent),

            "delta_cv_rmse": float(delta_cv_rmse),
            "delta_test_rmse": float(delta_test_rmse)
        }

        append_row_to_csv(out_csv, row_dict)
        print(f"Saved row -> {out_csv}")

    print(f"\n--- Regression Horizon {day} Day(s) ---\n")
    print(tabulate(
        rows,
        headers=[
            "Model",
            "CV RMSE (NoSent)",
            "CV RMSE (Sent)",
            "Test RMSE (NoSent)",
            "Test RMSE (Sent)",
            "Δ Test RMSE"
        ],
        tablefmt="github"
    ))