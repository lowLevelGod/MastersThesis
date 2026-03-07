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
df = pd.read_parquet("us_market_dataset_sector_and_industry_adjusted_returns.parquet")

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

CLASSIFICATION_LABEL_COLS = [[f"label_{day}d", f"label_{day}d_sector_adj", f"label_{day}d_industry_adj"] for day in range(1, 8)]
CLASSIFICATION_LABEL_COLS = [col for sublist in CLASSIFICATION_LABEL_COLS for col in sublist]
REGRESSION_LABEL_COLS = [[f"return_{day}d", f"return_{day}d_sector_adj", f"return_{day}d_industry_adj"] for day in range(1, 8)]
REGRESSION_LABEL_COLS = [col for sublist in REGRESSION_LABEL_COLS for col in sublist]

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

# ============================================================
# FULL DATASET EXPERIMENT
# ============================================================

def run_full_dataset_experiment(
    df,
    models_dict,
    task,
    label_type="raw",   # "raw", "sector_adj", "industry_adj"
    results_prefix="full_results"
):

    print("\n" + "="*80)
    print(f"RUNNING FULL DATASET | {task.upper()} | LABEL TYPE: {label_type}")
    print("="*80)

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    X_with_sent = build_feature_matrix(df, DROP_COLS, False, SENTIMENT_COLS)
    X_no_sent = build_feature_matrix(df, DROP_COLS, True, SENTIMENT_COLS)

    splitter = ExpandingWindowSplitter(n_splits=5, test_ratio=0.2)
    train_val_df, test_df, tscv = splitter.split(df, DATE_COL)

    X_train_val_with = X_with_sent.loc[train_val_df.index]
    X_test_with = X_with_sent.loc[test_df.index]
    X_train_val_no = X_no_sent.loc[train_val_df.index]
    X_test_no = X_no_sent.loc[test_df.index]

    for model_name, searcher in models_dict.items():

        csv_path = f"{results_prefix}_{task}_{label_type}_{model_name}.csv"

        if os.path.exists(csv_path):
            continue

        for day in range(1, 8):

            # --------------------------------------------
            # SELECT CORRECT LABEL
            # --------------------------------------------

            if task == "classification":

                if label_type == "raw":
                    label_col = f"label_{day}d"
                elif label_type == "sector_adj":
                    label_col = f"label_{day}d_sector_adj"
                else:
                    label_col = f"label_{day}d_industry_adj"

                if label_col not in df.columns:
                    continue

                y_train_val = df.loc[train_val_df.index, label_col].astype(int)
                y_test = df.loc[test_df.index, label_col].astype(int)

            else:

                if label_type == "raw":
                    label_col = f"return_{day}d"
                elif label_type == "sector_adj":
                    label_col = f"return_{day}d_sector_adj"
                else:
                    label_col = f"return_{day}d_industry_adj"

                if label_col not in df.columns:
                    continue

                y_train_val = df.loc[train_val_df.index, label_col].astype(float)
                y_test = df.loc[test_df.index, label_col].astype(float)

            # --------------------------------------------
            # GRID SEARCH
            # --------------------------------------------

            best_params_no, _ = searcher.run(X_train_val_no, y_train_val, tscv)
            best_params_sent, _ = searcher.run(X_train_val_with, y_train_val, tscv)

            # --------------------------------------------
            # FINAL FIT
            # --------------------------------------------

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

            # --------------------------------------------
            # EVALUATION
            # --------------------------------------------

            if task == "classification":

                _, score_no = trainer_no.evaluate(X_test_no, y_test)
                _, score_sent = trainer_sent.evaluate(X_test_with, y_test)

                delta = score_sent - score_no

                baseline_class = y_train_val.mode()[0]
                baseline_preds = np.full_like(y_test, baseline_class)
                baseline_score = f1_score(y_test, baseline_preds, average="macro")

            else:

                _, score_no, r2_no = trainer_no.evaluate(X_test_no, y_test)
                _, score_sent, r2_sent = trainer_sent.evaluate(X_test_with, y_test)

                delta = score_sent - score_no

                baseline_val = y_train_val.mean()
                baseline_preds = np.full_like(y_test, baseline_val)
                baseline_score = np.sqrt(mean_squared_error(y_test, baseline_preds))

            row_dict = {
                "horizon_days": day,
                "samples": len(df),
                "test_no_sent": score_no,
                "test_with_sent": score_sent,
                "delta": delta,
                "baseline": baseline_score
            }

            append_row_to_csv(csv_path, row_dict)

        print(f"Finished {task} | {label_type} | {model_name}")
        
        
# RAW
# run_full_dataset_experiment(df, classification_models, "classification", "raw")
# run_full_dataset_experiment(df, regression_models, "regression", "raw")

# # SECTOR-ADJUSTED
# run_full_dataset_experiment(df, classification_models, "classification", "sector_adj")
# run_full_dataset_experiment(df, regression_models, "regression", "sector_adj")

# # INDUSTRY-ADJUSTED
# run_full_dataset_experiment(df, classification_models, "classification", "industry_adj")
# run_full_dataset_experiment(df, regression_models, "regression", "industry_adj")

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

def analyze_full_results(task, models_dict):
    set_publication_style()
    
    label_types = ["raw", "sector_adj", "industry_adj"]

    for model_name in models_dict.keys():

        print("\n" + "="*80)
        print(f"{task.upper()} | MODEL: {model_name}")
        print("="*80)

        all_results = []

        for label_type in label_types:

            csv_path = f"full_results_{task}_{label_type}_{model_name}.csv"
            if not os.path.exists(csv_path):
                continue

            df_res = pd.read_csv(csv_path)
            df_res["label_type"] = label_type
            all_results.append(df_res)

        if not all_results:
            continue

        df_all = pd.concat(all_results)

        # =====================================================
        # 1️⃣ Plot delta per horizon
        # =====================================================

        sns.lineplot(
            data=df_all,
            x="horizon_days",
            y="delta",
            hue="label_type",
            marker="o"
        )

        plt.axhline(0, linestyle="--")
        plt.title(f"{task} - Sentiment Gain per Horizon - {model_name}")
        plt.tight_layout()
        plt.savefig(f"comparison_{task}_{model_name}.pdf")
        plt.close()

        # =====================================================
        # 2️⃣ Statistical Testing Across Horizons
        # =====================================================

        print("\n--- Statistical Inference Across Horizons ---")

        for label_type in df_all["label_type"].unique():

            subset = df_all[df_all["label_type"] == label_type]
            deltas = subset["delta"].dropna().values

            if len(deltas) < 3:
                continue

            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas, ddof=1)
            n = len(deltas)

            ci_low = mean_delta - 1.96 * std_delta / np.sqrt(n)
            ci_high = mean_delta + 1.96 * std_delta / np.sqrt(n)

            t_stat, p_val = ttest_1samp(deltas, 0)

            try:
                _, w_p = wilcoxon(deltas)
            except:
                w_p = np.nan

            print(f"\nLabel type: {label_type}")
            print(f"N horizons: {n}")
            print(f"Mean Δ: {mean_delta:.6f}")
            print(f"95% CI: ({ci_low:.6f}, {ci_high:.6f})")
            print(f"T-test p-value: {p_val:.6f}")
            print(f"Wilcoxon p-value: {w_p}")

        # =====================================================
        # 3️⃣ Baseline Comparison (Model vs Naive)
        # =====================================================

        print("\n--- Baseline Comparison ---")

        for label_type in df_all["label_type"].unique():

            subset = df_all[df_all["label_type"] == label_type]

            if task == "classification":
                improvement_no = subset["test_no_sent"] - subset["baseline"]
                improvement_sent = subset["test_with_sent"] - subset["baseline"]
            else:
                # Lower RMSE is better
                improvement_no = subset["baseline"] - subset["test_no_sent"]
                improvement_sent = subset["baseline"] - subset["test_with_sent"]

            mean_no = improvement_no.mean()
            mean_sent = improvement_sent.mean()

            print(f"\nLabel type: {label_type}")
            print(f"Mean improvement over baseline (NO sentiment): {mean_no:.6f}")
            print(f"Mean improvement over baseline (WITH sentiment): {mean_sent:.6f}")

            # Test if sentiment model beats baseline significantly
            t_stat, p_val = ttest_1samp(improvement_sent, 0)

            print(f"T-test (sentiment vs baseline=0) p-value: {p_val:.6f}")

        # =====================================================
        # 4️⃣ Raw vs Adjusted Comparison
        # =====================================================

        if set(["raw","sector_adj"]).issubset(df_all["label_type"].unique()):

            raw = df_all[df_all["label_type"]=="raw"]["delta"].values
            sector = df_all[df_all["label_type"]=="sector_adj"]["delta"].values

            if len(raw)==len(sector):

                diff = raw - sector
                t_stat, p_val = ttest_1samp(diff, 0)

                print("\n--- RAW vs SECTOR-ADJUSTED ---")
                print("Mean difference:", diff.mean())
                print("p-value:", p_val)

        print("\nAnalysis completed for:", model_name)
        
analyze_full_results("classification", classification_models)
analyze_full_results("regression", regression_models)