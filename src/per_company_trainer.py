import os
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

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
        )
        
class RandomForestClassifierGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="classification")

    def get_param_grid(self):
        return {
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "min_samples_leaf": [3, 5]
        }

    def build_model(self, params):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )
        
class XGBoostClassifierGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="classification")

    def get_param_grid(self):
        return {
            "n_estimators": [50],
            "max_depth": [2],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [10]
        }

    def build_model(self, params):
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            objective="multi:softmax",
            eval_metric="mlogloss",
            num_class = 3,
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
    
class RandomForestRegressorGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="regression")

    def get_param_grid(self):
        return {
            "n_estimators": [50, 100],
            "max_depth": [2, 3],
            "min_samples_leaf": [3, 5]
        }

    def build_model(self, params):
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )
        
class XGBoostRegressorGridSearch(BaseGridSearch):
    def __init__(self):
        super().__init__(task="regression")

    def get_param_grid(self):
        return {
            "n_estimators": [50],
            "max_depth": [2],
            "learning_rate": [0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [10]
        }

    def build_model(self, params):
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
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
df = pd.read_parquet("us_market_dataset.parquet")

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
# Per-Company Experiment (ENHANCED)
# ============================================================

MIN_SAMPLES_PER_COMPANY = 60
COMPANY_RESULTS_CLASSIFICATION_CSV = "per_company_results_classification.csv"
COMPANY_RESULTS_REGRESSION_CSV = "per_company_results_regression.csv"

# if os.path.exists(COMPANY_RESULTS_CLASSIFICATION_CSV):
#     os.remove(COMPANY_RESULTS_CLASSIFICATION_CSV)

# if os.path.exists(COMPANY_RESULTS_REGRESSION_CSV):
#     os.remove(COMPANY_RESULTS_REGRESSION_CSV)
    
tickers = df["ticker"].dropna().unique()
print(f"\nTotal tickers found: {len(tickers)}")

classification_models = {
    "Logistic": LogisticRegressionGridSearch(),
    "RandomForest": RandomForestClassifierGridSearch(),
    "XGBoost": XGBoostClassifierGridSearch()
}

regression_models = {
    "Ridge": RidgeGridSearch(),
    "RandomForest": RandomForestRegressorGridSearch(),
    "XGBoost": XGBoostRegressorGridSearch()
}

# ======================================================
# CLASSIFICATION (PER MODEL)
# ======================================================

for model_name, searcher in classification_models.items():

    COMPANY_RESULTS_CLASSIFICATION_CSV = f"per_company_results_classification_{model_name}.csv"

    if os.path.exists(COMPANY_RESULTS_CLASSIFICATION_CSV):
        # os.remove(COMPANY_RESULTS_CLASSIFICATION_CSV)
        continue

    for ticker in tickers:

        df_company = df[df["ticker"] == ticker].copy()

        if len(df_company) < MIN_SAMPLES_PER_COMPANY:
            continue

        df_company = df_company.sort_values(DATE_COL).reset_index(drop=True)

        X_with_sent = build_feature_matrix(df_company, DROP_COLS, False, SENTIMENT_COLS)
        X_no_sent = build_feature_matrix(df_company, DROP_COLS, True, SENTIMENT_COLS)

        splitter = ExpandingWindowSplitter(n_splits=3, test_ratio=0.2)
        train_val_df, test_df, tscv = splitter.split(df_company, DATE_COL)

        X_train_val_with = X_with_sent.loc[train_val_df.index]
        X_test_with = X_with_sent.loc[test_df.index]
        X_train_val_no = X_no_sent.loc[train_val_df.index]
        X_test_no = X_no_sent.loc[test_df.index]

        for day in range(1, 8):

            label_col = f"label_{day}d"
            if label_col not in df_company.columns:
                continue
            
            df_company[label_col] = df[label_col] - df[label_col].min()
            y_train_val = df_company.loc[train_val_df.index, label_col].astype(int)
            y_test = df_company.loc[test_df.index, label_col].astype(int)

            best_params_no, _ = searcher.run(X_train_val_no, y_train_val, tscv)
            best_params_sent, _ = searcher.run(X_train_val_with, y_train_val, tscv)

            fold_f1_no = []
            fold_f1_sent = []

            for train_idx, val_idx in tscv.split(X_train_val_no):

                X_tr_no, X_val_no = X_train_val_no.iloc[train_idx], X_train_val_no.iloc[val_idx]
                X_tr_sent, X_val_sent = X_train_val_with.iloc[train_idx], X_train_val_with.iloc[val_idx]
                y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

                trainer_no = ModelTrainer(
                    searcher.build_model(best_params_no),
                    task="classification",
                    scale=searcher.scale_required()
                )
                trainer_no.fit(X_tr_no, y_tr)
                preds_no = trainer_no.pipeline.predict(X_val_no)
                fold_f1_no.append(f1_score(y_val, preds_no, average="macro"))

                trainer_sent = ModelTrainer(
                    searcher.build_model(best_params_sent),
                    task="classification",
                    scale=searcher.scale_required()
                )
                trainer_sent.fit(X_tr_sent, y_tr)
                preds_sent = trainer_sent.pipeline.predict(X_val_sent)
                fold_f1_sent.append(f1_score(y_val, preds_sent, average="macro"))

            mean_no = np.mean(fold_f1_no)
            mean_sent = np.mean(fold_f1_sent)
            ci_no = 1.96 * np.std(fold_f1_no) / np.sqrt(len(fold_f1_no))
            ci_sent = 1.96 * np.std(fold_f1_sent) / np.sqrt(len(fold_f1_sent))
            _, p_value = ttest_rel(fold_f1_sent, fold_f1_no)

            trainer_no = ModelTrainer(
                searcher.build_model(best_params_no),
                task="classification",
                scale=searcher.scale_required()
            )
            trainer_no.fit(X_train_val_no, y_train_val)
            _, test_f1_no = trainer_no.evaluate(X_test_no, y_test)

            trainer_sent = ModelTrainer(
                searcher.build_model(best_params_sent),
                task="classification",
                scale=searcher.scale_required()
            )
            trainer_sent.fit(X_train_val_with, y_train_val)
            _, test_f1_sent = trainer_sent.evaluate(X_test_with, y_test)

            delta_test_f1 = test_f1_sent - test_f1_no

            majority_class = y_train_val.mode()[0]
            baseline_preds = np.full_like(y_test, majority_class)
            baseline_f1 = f1_score(y_test, baseline_preds, average="macro")

            coef_magnitude = np.nan
            model_obj = trainer_sent.pipeline.named_steps["model"]
            if hasattr(model_obj, "coef_"):
                coef_magnitude = np.mean(np.abs(model_obj.coef_))

            row_dict = {
                "ticker": ticker,
                "horizon_days": day,
                "samples": len(df_company),
                "cv_mean_no_sent": mean_no,
                "cv_mean_with_sent": mean_sent,
                "cv_ci_no_sent": ci_no,
                "cv_ci_with_sent": ci_sent,
                "p_value": p_value,
                "test_f1_no_sent": test_f1_no,
                "test_f1_with_sent": test_f1_sent,
                "delta_test_f1": delta_test_f1,
                "baseline_f1": baseline_f1,
                "coef_magnitude": coef_magnitude,
            }

            append_row_to_csv(COMPANY_RESULTS_CLASSIFICATION_CSV, row_dict)

    print(f"\nFinished classification for model: {model_name}")

# ======================================================
# REGRESSION (ENHANCED)
# ======================================================
for model_name, searcher in regression_models.items():

    COMPANY_RESULTS_REGRESSION_CSV = f"per_company_results_regression_{model_name}.csv"

    if os.path.exists(COMPANY_RESULTS_REGRESSION_CSV):
        # os.remove(COMPANY_RESULTS_REGRESSION_CSV)
        continue
    
    for ticker in tickers:

        df_company = df[df["ticker"] == ticker].copy()

        if len(df_company) < MIN_SAMPLES_PER_COMPANY:
            continue

        df_company = df_company.sort_values(DATE_COL).reset_index(drop=True)

        X_with_sent = build_feature_matrix(df_company, DROP_COLS, False, SENTIMENT_COLS)
        X_no_sent = build_feature_matrix(df_company, DROP_COLS, True, SENTIMENT_COLS)

        splitter = ExpandingWindowSplitter(n_splits=3, test_ratio=0.2)
        train_val_df, test_df, tscv = splitter.split(df_company, DATE_COL)

        X_train_val_with = X_with_sent.loc[train_val_df.index]
        X_test_with = X_with_sent.loc[test_df.index]
        X_train_val_no = X_no_sent.loc[train_val_df.index]
        X_test_no = X_no_sent.loc[test_df.index]
        
        for day in range(1, 8):    
            target_col = f"return_{day}d"
            if target_col in df_company.columns:
                y_train_val = df_company.loc[train_val_df.index, target_col].astype(float)
                y_test = df_company.loc[test_df.index, target_col].astype(float)

                best_params_no, _ = searcher.run(X_train_val_no, y_train_val, tscv)
                best_params_sent, _ = searcher.run(X_train_val_with, y_train_val, tscv)

                # --------------------------------------------------
                # Fold-level RMSE
                # --------------------------------------------------
                fold_rmse_no = []
                fold_rmse_sent = []

                for train_idx, val_idx in tscv.split(X_train_val_no):

                    X_tr_no, X_val_no = X_train_val_no.iloc[train_idx], X_train_val_no.iloc[val_idx]
                    X_tr_sent, X_val_sent = X_train_val_with.iloc[train_idx], X_train_val_with.iloc[val_idx]
                    y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

                    trainer_no = ModelTrainer(
                        searcher.build_model(best_params_no),
                        task="regression",
                        scale=True
                    )
                    trainer_no.fit(X_tr_no, y_tr)
                    preds_no = trainer_no.pipeline.predict(X_val_no)
                    rmse_no = np.sqrt(mean_squared_error(y_val, preds_no))
                    fold_rmse_no.append(rmse_no)

                    trainer_sent = ModelTrainer(
                        searcher.build_model(best_params_sent),
                        task="regression",
                        scale=True
                    )
                    trainer_sent.fit(X_tr_sent, y_tr)
                    preds_sent = trainer_sent.pipeline.predict(X_val_sent)
                    rmse_sent = np.sqrt(mean_squared_error(y_val, preds_sent))
                    fold_rmse_sent.append(rmse_sent)

                mean_rmse_no = np.mean(fold_rmse_no)
                mean_rmse_sent = np.mean(fold_rmse_sent)

                std_rmse_no = np.std(fold_rmse_no)
                std_rmse_sent = np.std(fold_rmse_sent)

                ci_rmse_no = 1.96 * std_rmse_no / np.sqrt(len(fold_rmse_no))
                ci_rmse_sent = 1.96 * std_rmse_sent / np.sqrt(len(fold_rmse_sent))

                t_stat_rmse, p_value_rmse = ttest_rel(fold_rmse_sent, fold_rmse_no)

                # --------------------------------------------------
                # Final fit
                # --------------------------------------------------
                trainer_no = ModelTrainer(
                    searcher.build_model(best_params_no),
                    task="regression",
                    scale=True
                )
                trainer_no.fit(X_train_val_no, y_train_val)
                mae_no, test_rmse_no, r2_no = trainer_no.evaluate(X_test_no, y_test)

                trainer_sent = ModelTrainer(
                    searcher.build_model(best_params_sent),
                    task="regression",
                    scale=True
                )
                trainer_sent.fit(X_train_val_with, y_train_val)
                mae_sent, test_rmse_sent, r2_sent = trainer_sent.evaluate(X_test_with, y_test)

                delta_test_rmse = test_rmse_sent - test_rmse_no
                delta_r2 = r2_sent - r2_no

                # --------------------------------------------------
                # Baseline (mean predictor)
                # --------------------------------------------------
                baseline_value = y_train_val.mean()
                baseline_preds = np.full_like(y_test, baseline_value)
                baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

                # --------------------------------------------------
                # Residual variance
                # --------------------------------------------------
                residuals = y_test - trainer_sent.pipeline.predict(X_test_with)
                residual_variance = np.var(residuals)

                # --------------------------------------------------
                # Prediction correlation
                # --------------------------------------------------
                pred_corr = np.corrcoef(
                    trainer_sent.pipeline.predict(X_test_with),
                    y_test
                )[0, 1]

                row_dict = {
                    "ticker": ticker,
                    "task": "regression",
                    "horizon_days": day,
                    "samples": len(df_company),

                    "cv_rmse_no_sent": mean_rmse_no,
                    "cv_rmse_with_sent": mean_rmse_sent,
                    "cv_ci_rmse_no": ci_rmse_no,
                    "cv_ci_rmse_with": ci_rmse_sent,
                    "p_value_rmse": p_value_rmse,

                    "test_rmse_no_sent": test_rmse_no,
                    "test_rmse_with_sent": test_rmse_sent,
                    "delta_test_rmse": delta_test_rmse,
                    "delta_r2": delta_r2,

                    "baseline_rmse": baseline_rmse,
                    "residual_variance": residual_variance,
                    "prediction_correlation": pred_corr
                }

                append_row_to_csv(COMPANY_RESULTS_REGRESSION_CSV, row_dict)

    print(f"\nFinished regression for model: {model_name}")
    
for model_name in classification_models.keys():

    csv_path = f"per_company_results_classification_{model_name}.csv"
    if not os.path.exists(csv_path):
        continue

    results = pd.read_csv(csv_path)

    print(f"\n===== Classification Results: {model_name} =====")
    print(results["delta_test_f1"].describe())

    sns.histplot(results["delta_test_f1"].dropna(), bins=30)
    plt.title(f"ΔF1 Distribution - {model_name}")
    plt.savefig(f"distribution_sent_{model_name}.pdf")
    plt.close()

    plt.scatter(results["samples"], results["delta_test_f1"])
    plt.xlabel("Samples")
    plt.ylabel("ΔF1")
    plt.title(f"Samples vs ΔF1 - {model_name}")
    plt.savefig(f"samples_vs_deltaF1_{model_name}.pdf")
    plt.close()

    sig_rate = (results["p_value"] < 0.05).mean()
    print("Significant %:", sig_rate)

    results["beats_baseline"] = results["test_f1_with_sent"] > results["baseline_f1"]
    print("Beats baseline %:", results["beats_baseline"].mean())
    
    
for model_name in regression_models.keys():

    csv_path = f"per_company_results_regression_{model_name}.csv"
    if not os.path.exists(csv_path):
        continue

    reg_results = pd.read_csv(csv_path)

    print(f"\n===== Regression Results: {model_name} =====")
    print(reg_results["delta_test_rmse"].describe())

    sns.histplot(reg_results["delta_test_rmse"].dropna(), bins=30)
    plt.title(f"ΔRMSE Distribution - {model_name}")
    plt.savefig(f"distribution_rmse_{model_name}.pdf")
    plt.close()

    plt.scatter(reg_results["samples"], reg_results["delta_test_rmse"])
    plt.xlabel("Samples")
    plt.ylabel("ΔRMSE")
    plt.title(f"Samples vs ΔRMSE - {model_name}")
    plt.savefig(f"samples_vs_deltaRMSE_{model_name}.pdf")
    plt.close()

    sig_rate = (reg_results["p_value_rmse"] < 0.05).mean()
    print("Significant %:", sig_rate)

    reg_results["beats_baseline"] = (
        reg_results["test_rmse_with_sent"] < reg_results["baseline_rmse"]
    )
    print("Beats baseline %:", reg_results["beats_baseline"].mean())