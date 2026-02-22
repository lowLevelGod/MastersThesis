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
df = pd.read_parquet("us_market_dataset_with_sector.parquet")

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

def run_group_experiment(
    df,
    group_col,
    models_dict,
    task,
    min_samples=200,
    results_prefix="results"
):
    """
    Generic training loop for grouping column:
    group_col = 'ticker' | 'sector' | 'industry'
    """

    groups = df[group_col].dropna().unique()
    print(f"\nRunning experiment grouped by: {group_col}")
    print(f"Total groups found: {len(groups)}")

    for model_name, searcher in models_dict.items():

        csv_path = f"{results_prefix}_{group_col}_{task}_{model_name}.csv"

        if os.path.exists(csv_path):
            continue

        for group in groups:
            
            if group_col == "ticker" and group == "XRX":
                continue

            df_group = df[df[group_col] == group].copy()

            if len(df_group) < min_samples:
                continue

            df_group = df_group.sort_values(DATE_COL).reset_index(drop=True)

            X_with_sent = build_feature_matrix(df_group, DROP_COLS, False, SENTIMENT_COLS)
            X_no_sent = build_feature_matrix(df_group, DROP_COLS, True, SENTIMENT_COLS)

            splitter = ExpandingWindowSplitter(n_splits=3, test_ratio=0.2)
            train_val_df, test_df, tscv = splitter.split(df_group, DATE_COL)

            X_train_val_with = X_with_sent.loc[train_val_df.index]
            X_test_with = X_with_sent.loc[test_df.index]
            X_train_val_no = X_no_sent.loc[train_val_df.index]
            X_test_no = X_no_sent.loc[test_df.index]

            for day in range(1, 8):

                if task == "classification":
                    label_col = f"label_{day}d"
                    if label_col not in df_group.columns:
                        continue
                    
                    y_train_val = df_group.loc[train_val_df.index, label_col].astype(int)
                    y_test = df_group.loc[test_df.index, label_col].astype(int)

                else:
                    label_col = f"return_{day}d"
                    if label_col not in df_group.columns:
                        continue

                    y_train_val = df_group.loc[train_val_df.index, label_col].astype(float)
                    y_test = df_group.loc[test_df.index, label_col].astype(float)

                # -------------------------------------------------
                # Grid Search
                # -------------------------------------------------
                best_params_no, _ = searcher.run(X_train_val_no, y_train_val, tscv)
                best_params_sent, _ = searcher.run(X_train_val_with, y_train_val, tscv)

                # -------------------------------------------------
                # Final Training
                # -------------------------------------------------
                trainer_no = ModelTrainer(
                    searcher.build_model(best_params_no),
                    task=task,
                    scale=searcher.scale_required()
                )
                trainer_no.fit(X_train_val_no, y_train_val)

                trainer_sent = ModelTrainer(
                    searcher.build_model(best_params_sent),
                    task=task,
                    scale=searcher.scale_required()
                )
                trainer_sent.fit(X_train_val_with, y_train_val)

                # -------------------------------------------------
                # Evaluation
                # -------------------------------------------------
                if task == "classification":

                    _, f1_no = trainer_no.evaluate(X_test_no, y_test)
                    _, f1_sent = trainer_sent.evaluate(X_test_with, y_test)

                    delta = f1_sent - f1_no

                    baseline_class = y_train_val.mode()[0]
                    baseline_preds = np.full_like(y_test, baseline_class)
                    baseline_score = f1_score(y_test, baseline_preds, average="macro")

                    row_dict = {
                        group_col: group,
                        "horizon_days": day,
                        "samples": len(df_group),
                        "test_no_sent": f1_no,
                        "test_with_sent": f1_sent,
                        "delta": delta,
                        "baseline": baseline_score
                    }

                else:

                    _, rmse_no, r2_no = trainer_no.evaluate(X_test_no, y_test)
                    _, rmse_sent, r2_sent = trainer_sent.evaluate(X_test_with, y_test)

                    delta = rmse_sent - rmse_no

                    baseline_val = y_train_val.mean()
                    baseline_preds = np.full_like(y_test, baseline_val)
                    baseline_score = np.sqrt(mean_squared_error(y_test, baseline_preds))

                    row_dict = {
                        group_col: group,
                        "horizon_days": day,
                        "samples": len(df_group),
                        "test_no_sent": rmse_no,
                        "test_with_sent": rmse_sent,
                        "delta": delta,
                        "baseline": baseline_score,
                        "delta_r2": r2_sent - r2_no
                    }

                append_row_to_csv(csv_path, row_dict)

        print(f"Finished {task} for {group_col} - Model: {model_name}")
        

# -------------------------
# SECTOR TRAINING
# -------------------------
# run_group_experiment(
#     df=df,
#     group_col="sector",
#     models_dict=classification_models,
#     task="classification",
#     min_samples=500,
#     results_prefix="group_results"
# )

# run_group_experiment(
#     df=df,
#     group_col="sector",
#     models_dict=regression_models,
#     task="regression",
#     min_samples=500,
#     results_prefix="group_results"
# )

# -------------------------
# INDUSTRY TRAINING
# -------------------------
# run_group_experiment(
#     df=df,
#     group_col="industry",
#     models_dict=classification_models,
#     task="classification",
#     min_samples=300,
#     results_prefix="group_results"
# )

# run_group_experiment(
#     df=df,
#     group_col="industry",
#     models_dict=regression_models,
#     task="regression",
#     min_samples=300,
#     results_prefix="group_results"
# )

# -----------------------
# COMPANY TRAINING
# -----------------------
run_group_experiment(
    df=df,
    group_col="ticker",
    models_dict=classification_models,
    task="classification",
    min_samples=60,
    results_prefix="group_results"
)

run_group_experiment(
    df=df,
    group_col="ticker",
    models_dict=regression_models,
    task="regression",
    min_samples=60,
    results_prefix="group_results"
)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, wilcoxon

def analyze_group_results(group_col, task, models_dict):

    for model_name in models_dict.keys():

        csv_path = f"group_results_{group_col}_{task}_{model_name}.csv"
        if not os.path.exists(csv_path):
            continue

        df_res = pd.read_csv(csv_path)

        print("\n" + "="*70)
        print(f"{group_col.upper()} | {task.upper()} | {model_name}")
        print("="*70)

        # -------------------------------------------------------
        # 1️⃣ Basic distribution plot (ALL deltas, descriptive only)
        # -------------------------------------------------------

        sns.histplot(df_res["delta"].dropna(), bins=30)
        plt.title(f"{group_col} Δ Distribution - {model_name}")
        plt.xlabel("Delta")
        plt.tight_layout()
        plt.savefig(f"{group_col}_{task}_delta_distribution_{model_name}.pdf")
        plt.close()

        # -------------------------------------------------------
        # 2️⃣ PER-HORIZON ANALYSIS (STATISTICALLY CORRECT)
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

            cohens_d = mean_delta / std_delta if std_delta > 0 else np.nan

            print(f"\nHorizon {h} days")
            print(f"N groups: {n}")
            print(f"Mean Δ: {mean_delta:.6f}")
            print(f"95% CI: ({ci_low:.6f}, {ci_high:.6f})")
            print(f"T-test p-value: {p_val:.6f}")
            print(f"Wilcoxon p-value: {w_p}")
            print(f"Cohen's d: {cohens_d:.4f}")

        # -------------------------------------------------------
        # 3️⃣ OVERALL EFFECT (AVERAGE PER GROUP FIRST)
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

            cohens_d = mean_delta / std_delta if std_delta > 0 else np.nan

            print(f"N groups: {n}")
            print(f"Mean Δ (avg across horizons): {mean_delta:.6f}")
            print(f"95% CI: ({ci_low:.6f}, {ci_high:.6f})")
            print(f"T-test p-value: {p_val:.6f}")
            print(f"Wilcoxon p-value: {w_p}")
            print(f"Cohen's d: {cohens_d:.4f}")

        # -------------------------------------------------------
        # 4️⃣ WHO BENEFITTED MOST?
        # -------------------------------------------------------

        print("\n--- Top & Bottom Performing Groups ---")

        ranked = avg_per_group.sort_values(ascending=False)

        print("\nTop 5 groups (most positive Δ):")
        print(ranked.head(5))

        print("\nBottom 5 groups (most negative Δ):")
        print(ranked.tail(5))

        # -------------------------------------------------------
        # 5️⃣ Baseline Comparison
        # -------------------------------------------------------

        if task == "classification":
            beats = (df_res["test_with_sent"] > df_res["baseline"]).mean()
        else:
            beats = (df_res["test_with_sent"] < df_res["baseline"]).mean()

        print("\nBeats baseline percentage:", round(beats*100, 2), "%")

        # -------------------------------------------------------
        # 6️⃣ Sample Size vs Improvement Relationship
        # -------------------------------------------------------

        avg_samples = (
            df_res.groupby(group_col)["samples"]
            .mean()
        )

        merged = pd.DataFrame({
            "avg_delta": avg_per_group,
            "avg_samples": avg_samples
        }).dropna()

        corr = merged["avg_delta"].corr(merged["avg_samples"])

        print("Correlation between sample size and Δ:", round(corr, 4))

        plt.scatter(merged["avg_samples"], merged["avg_delta"])
        plt.xlabel("Average Samples")
        plt.ylabel("Average Δ")
        plt.title(f"{group_col}: Samples vs Sentiment Gain")
        plt.tight_layout()
        plt.savefig(f"{group_col}_{task}_samples_vs_avg_delta_{model_name}.pdf")
        plt.close()

        print("\nAnalysis completed for:", model_name)
        
        
# analyze_group_results("sector", "classification", classification_models)
# analyze_group_results("sector", "regression", regression_models)

# analyze_group_results("industry", "classification", classification_models)
# analyze_group_results("industry", "regression", regression_models)

analyze_group_results("ticker", "classification", classification_models)
analyze_group_results("ticker", "regression", regression_models)