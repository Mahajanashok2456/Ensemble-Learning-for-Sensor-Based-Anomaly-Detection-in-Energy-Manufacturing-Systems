# ============================================================
# BINARY CLASSIFICATION PIPELINE
# Dataset: train.parquet / test.parquet
# Target: binary 0/1, severely imbalanced (99.14% class 0)
# Models: LightGBM + XGBoost with 5-fold stratified CV
# ============================================================

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score,
)

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 0: PATHS AND GLOBAL CONFIGURATION
# ============================================================

BASE_DIR    = r"c:\Users\MAHAJAN ASHOK\OneDrive\Desktop\sure trust\project"
TRAIN_PATH  = os.path.join(BASE_DIR, "train.parquet")
TEST_PATH   = os.path.join(BASE_DIR, "test.parquet")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
SUBMIT_PATH = os.path.join(BASE_DIR, "submissions.csv")

RANDOM_STATE     = 42
N_FOLDS          = 5
SCALE_POS_WEIGHT = 115.78  # class_0_count / class_1_count confirmed from data

np.random.seed(RANDOM_STATE)

# Global reference to test set so CV loop can generate test predictions per fold
df_test_global = None

# Feature columns (populated after feature engineering in Section 3)
FEATURE_COLS = [
    # Raw continuous features
    "X1", "X2",
    # Discrete raw (kept as tree fallback)
    "X3", "X4", "X5",
    # Log/exp decoded - HIGH SIGNAL
    "X3_log", "X4_log", "X5_int",
    # Date-derived numerics
    "year", "month", "day", "day_of_week", "quarter", "day_of_year",
    "is_month_end", "is_month_start", "is_weekend",
    # Cyclical encoding of periodic features
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Interaction features
    "X1_minus_X2", "X1_X2_ratio", "X3log_X4log",
]

LGBM_PARAMS = {
    "n_estimators"     : 2000,
    "learning_rate"    : 0.05,
    "num_leaves"       : 63,
    "max_depth"        : -1,
    "scale_pos_weight" : SCALE_POS_WEIGHT,
    "metric"           : "auc",          # use AUC for early stopping (binary_logloss converges at iter 1 with large scale_pos_weight)
    "min_child_samples": 50,
    "min_child_weight" : 1e-3,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "subsample"        : 0.8,
    "subsample_freq"   : 1,
    "colsample_bytree" : 0.8,
    "n_jobs"           : -1,
    "random_state"     : RANDOM_STATE,
    "verbose"          : -1,
}

XGBM_PARAMS = {
    "n_estimators"        : 2000,
    "learning_rate"       : 0.05,
    "max_depth"           : 6,
    "max_leaves"          : 63,
    "scale_pos_weight"    : SCALE_POS_WEIGHT,
    "min_child_weight"    : 5,
    "gamma"               : 0.1,
    "reg_alpha"           : 0.1,
    "reg_lambda"          : 1.0,
    "subsample"           : 0.8,
    "colsample_bytree"    : 0.8,
    "tree_method"         : "hist",
    "early_stopping_rounds": 50,
    "eval_metric"         : "aucpr",
    "n_jobs"              : -1,
    "random_state"        : RANDOM_STATE,
    "verbosity"           : 0,
}


# ============================================================
# SECTION 1: DATA LOADING
# ============================================================

def load_data():
    print("[1/8] Loading data...")
    start = time.time()

    df_train = pd.read_parquet(TRAIN_PATH)
    df_test  = pd.read_parquet(TEST_PATH)

    # Target is stored as string "0"/"1" — cast to int
    df_train["target"] = df_train["target"].astype(int)

    # Downcast float64 → float32: saves ~30 MB, no information loss
    float_cols = ["X1", "X2", "X3", "X4", "X5"]
    df_train[float_cols] = df_train[float_cols].astype(np.float32)
    df_test[float_cols]  = df_test[float_cols].astype(np.float32)

    elapsed = time.time() - start
    print(f"    Train: {df_train.shape}  Test: {df_test.shape}  ({elapsed:.1f}s)")
    return df_train, df_test


# ============================================================
# SECTION 2: EDA SUMMARY
# ============================================================

def run_eda(df_train, df_test):
    print("\n[2/8] Exploratory Data Analysis")
    print("=" * 60)

    print(f"Train shape : {df_train.shape}")
    print(f"Test  shape : {df_test.shape}")

    # Null check
    train_nulls = df_train.isnull().sum().sum()
    test_nulls  = df_test.isnull().sum().sum()
    print(f"\nNull values - Train: {train_nulls}  Test: {test_nulls}")

    # Class distribution
    counts = df_train["target"].value_counts().sort_index()
    total  = len(df_train)
    ratio  = counts[0] / counts[1]
    print(f"\nClass distribution:")
    print(f"  Class 0 : {counts[0]:>9,}  ({counts[0]/total*100:.2f}%)")
    print(f"  Class 1 : {counts[1]:>9,}  ({counts[1]/total*100:.2f}%)")
    print(f"  Imbalance ratio       : {ratio:.1f}:1")
    print(f"  scale_pos_weight used : {SCALE_POS_WEIGHT}")

    # Feature statistics
    feat_cols = ["X1", "X2", "X3", "X4", "X5"]
    print(f"\nFeature statistics (train):")
    print(df_train[feat_cols].astype(float).describe().round(4).to_string())

    # Pearson correlation with target
    print(f"\nPearson correlation with target:")
    for col in feat_cols:
        r = df_train[col].astype(float).corr(df_train["target"].astype(float))
        print(f"  {col} : {r:+.6f}")

    # Temporal target rate
    df_train["_year"] = df_train["Date"].dt.year
    yr_rates = df_train.groupby("_year")["target"].mean()
    print(f"\nTarget rate by year (temporal drift check):")
    for yr, rate in yr_rates.items():
        print(f"  {yr}: {rate*100:.3f}%")
    df_train.drop(columns=["_year"], inplace=True)

    # Date range
    print(f"\nDate range (train): {df_train['Date'].min().date()} to {df_train['Date'].max().date()}")
    print(f"Date range (test) : {df_test['Date'].min().date()} to {df_test['Date'].max().date()}")
    print("=" * 60)


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Applied identically to train and test.
    Key transformations:
    - X3, X4 are stored as e^n (integer n) → log-decode to int for 8-17x correlation gain
    - X5 is stored as ln(integer 1-32) → exp-decode to recover the integer
    - Date decomposed into cyclical and categorical time features
    """
    # Date features
    df["year"]           = df["Date"].dt.year.astype(np.int16)
    df["month"]          = df["Date"].dt.month.astype(np.int8)
    df["day"]            = df["Date"].dt.day.astype(np.int8)
    df["day_of_week"]    = df["Date"].dt.dayofweek.astype(np.int8)
    df["quarter"]        = df["Date"].dt.quarter.astype(np.int8)
    df["day_of_year"]    = df["Date"].dt.dayofyear.astype(np.int16)
    df["is_month_end"]   = df["Date"].dt.is_month_end.astype(np.int8)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(np.int8)
    df["is_weekend"]     = (df["Date"].dt.dayofweek >= 5).astype(np.int8)

    # Cyclical encoding — preserves circular proximity (e.g., Dec ↔ Jan)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)

    # X3 and X4 log-decode: correlation jumps from ~0.02/0.05 to ~0.33/0.38
    # np.log(e^n) = n; np.round handles float precision noise
    df["X3_log"] = np.round(np.log(df["X3"].clip(lower=1e-10))).astype(np.int16)
    df["X4_log"] = np.round(np.log(df["X4"].clip(lower=1e-10))).astype(np.int16)

    # X5 exp-decode: recovers integer 1-32
    df["X5_int"] = np.round(np.exp(df["X5"])).astype(np.int8)

    # Interaction features
    df["X1_minus_X2"] = (df["X1"] - df["X2"]).astype(np.float32)
    df["X1_X2_ratio"] = (df["X1"] / (df["X2"] + 1e-8)).astype(np.float32)
    df["X3log_X4log"] = (df["X3_log"].astype(np.int32) * df["X4_log"].astype(np.int32)).astype(np.int32)

    return df


# ============================================================
# SECTION 5: CROSS-VALIDATION TRAINING
# ============================================================

def train_with_cv(df_train):
    global df_test_global

    print("\n[5/8] Cross-Validation Training (5-fold Stratified)")

    X      = df_train[FEATURE_COLS].values
    y      = df_train["target"].values
    X_test = df_test_global[FEATURE_COLS].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds_lgbm  = np.zeros(len(df_train), dtype=np.float64)
    oof_preds_xgbm  = np.zeros(len(df_train), dtype=np.float64)
    test_preds_lgbm = np.zeros(len(df_test_global), dtype=np.float64)
    test_preds_xgbm = np.zeros(len(df_test_global), dtype=np.float64)

    lgbm_models = []
    xgbm_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS} ---")
        fold_start = time.time()

        X_tr, y_tr   = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        print(f"    Train: {len(X_tr):,}  Val: {len(X_val):,}"
              f"  Pos rate: {y_val.mean()*100:.3f}%")

        # ---------- LightGBM ----------
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        fold_lgbm_prob = lgbm_model.predict_proba(X_val)[:, 1]
        oof_preds_lgbm[val_idx] = fold_lgbm_prob

        auc_lgbm = roc_auc_score(y_val, fold_lgbm_prob)
        ap_lgbm  = average_precision_score(y_val, fold_lgbm_prob)
        print(f"    LGBM  AUC: {auc_lgbm:.4f}  AUPRC: {ap_lgbm:.4f}"
              f"  Best iter: {lgbm_model.best_iteration_}")

        test_preds_lgbm += lgbm_model.predict_proba(X_test)[:, 1] / N_FOLDS
        lgbm_models.append(lgbm_model)

        # ---------- XGBoost ----------
        xgbm_model = xgb.XGBClassifier(**XGBM_PARAMS)
        xgbm_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )
        fold_xgbm_prob = xgbm_model.predict_proba(X_val)[:, 1]
        oof_preds_xgbm[val_idx] = fold_xgbm_prob

        auc_xgbm = roc_auc_score(y_val, fold_xgbm_prob)
        ap_xgbm  = average_precision_score(y_val, fold_xgbm_prob)
        print(f"    XGBM  AUC: {auc_xgbm:.4f}  AUPRC: {ap_xgbm:.4f}"
              f"  Best iter: {xgbm_model.best_iteration}")

        test_preds_xgbm += xgbm_model.predict_proba(X_test)[:, 1] / N_FOLDS
        xgbm_models.append(xgbm_model)

        print(f"    Fold time: {(time.time()-fold_start)/60:.1f} min")

    return (oof_preds_lgbm, oof_preds_xgbm,
            test_preds_lgbm, test_preds_xgbm,
            lgbm_models, xgbm_models, y)


# ============================================================
# SECTION 6: EVALUATION AND THRESHOLD TUNING
# ============================================================

def evaluate_and_tune_threshold(y_true, oof_prob_lgbm, oof_prob_xgbm):
    print("\n[6/8] Evaluation and Threshold Tuning")
    results = {}

    for model_name, oof_prob in [("XGBoost", oof_prob_xgbm), ("LightGBM", oof_prob_lgbm)]:
        # Rank-order metrics (threshold-independent)
        auc_roc = roc_auc_score(y_true, oof_prob)
        auc_pr  = average_precision_score(y_true, oof_prob)

        # Sweep thresholds to find best F1
        thresholds = np.arange(0.01, 0.99, 0.005)
        f1_scores  = np.array([
            f1_score(y_true, (oof_prob >= t).astype(int), zero_division=0)
            for t in thresholds
        ])

        best_threshold = float(thresholds[f1_scores.argmax()])
        best_f1        = float(f1_scores.max())

        best_preds = (oof_prob >= best_threshold).astype(int)
        precision  = precision_score(y_true, best_preds, zero_division=0)
        recall     = recall_score(y_true, best_preds, zero_division=0)
        cm         = confusion_matrix(y_true, best_preds)

        print(f"\n  {model_name}:")
        print(f"    AUC-ROC (threshold-free) : {auc_roc:.4f}")
        print(f"    AUC-PR  (threshold-free) : {auc_pr:.4f}")
        print(f"    Best threshold for F1    : {best_threshold:.3f}")
        print(f"    F1  at best threshold    : {best_f1:.4f}")
        print(f"    Precision                : {precision:.4f}")
        print(f"    Recall                   : {recall:.4f}")
        print(f"    Confusion Matrix:\n{cm}")
        print(classification_report(y_true, best_preds,
                                    target_names=["class_0", "class_1"]))

        results[model_name] = {
            "auc_roc"       : round(auc_roc, 6),
            "auc_pr"        : round(auc_pr, 6),
            "best_threshold": round(best_threshold, 4),
            "best_f1"       : round(best_f1, 6),
            "precision"     : round(float(precision), 6),
            "recall"        : round(float(recall), 6),
        }

    # Weighted ensemble tuning over OOF probabilities
    print("\n  Ensemble tuning (weighted blend of XGBoost + LightGBM):")
    weight_grid = np.arange(0.0, 1.01, 0.05)  # weight for LightGBM
    thresholds = np.arange(0.01, 0.99, 0.005)

    best_ensemble = {
        "weight_lgbm": 0.5,
        "weight_xgbm": 0.5,
        "best_threshold": 0.5,
        "best_f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc_roc": 0.0,
        "auc_pr": 0.0,
    }

    for w_lgbm in weight_grid:
        w_xgbm = 1.0 - w_lgbm
        ensemble_oof = (w_lgbm * oof_prob_lgbm) + (w_xgbm * oof_prob_xgbm)

        auc_roc = roc_auc_score(y_true, ensemble_oof)
        auc_pr = average_precision_score(y_true, ensemble_oof)

        f1_scores = np.array([
            f1_score(y_true, (ensemble_oof >= t).astype(int), zero_division=0)
            for t in thresholds
        ])
        idx = int(f1_scores.argmax())
        best_threshold = float(thresholds[idx])
        best_f1 = float(f1_scores[idx])

        preds = (ensemble_oof >= best_threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)

        if best_f1 > best_ensemble["best_f1"]:
            best_ensemble = {
                "weight_lgbm": float(w_lgbm),
                "weight_xgbm": float(w_xgbm),
                "best_threshold": float(best_threshold),
                "best_f1": float(best_f1),
                "precision": float(precision),
                "recall": float(recall),
                "auc_roc": float(auc_roc),
                "auc_pr": float(auc_pr),
            }

    print(f"    Best weights -> LightGBM: {best_ensemble['weight_lgbm']:.2f}, XGBoost: {best_ensemble['weight_xgbm']:.2f}")
    print(f"    AUC-ROC: {best_ensemble['auc_roc']:.4f}  AUC-PR: {best_ensemble['auc_pr']:.4f}")
    print(f"    Best threshold: {best_ensemble['best_threshold']:.3f}  Best F1: {best_ensemble['best_f1']:.4f}")

    results["Ensemble"] = {
        "auc_roc": round(best_ensemble["auc_roc"], 6),
        "auc_pr": round(best_ensemble["auc_pr"], 6),
        "best_threshold": round(best_ensemble["best_threshold"], 4),
        "best_f1": round(best_ensemble["best_f1"], 6),
        "precision": round(best_ensemble["precision"], 6),
        "recall": round(best_ensemble["recall"], 6),
        "weight_lgbm": round(best_ensemble["weight_lgbm"], 4),
        "weight_xgbm": round(best_ensemble["weight_xgbm"], 4),
    }

    # Prefer ensemble as final primary model for pipeline/web inference
    primary = "Ensemble"
    print(f"\n  Primary model selected: {primary}")
    return results, primary


# ============================================================
# SECTION 7: SAVE MODELS
# ============================================================

def save_models(lgbm_models, xgbm_models, results):
    print("\n[7/8] Saving models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for i, model in enumerate(lgbm_models):
        path = os.path.join(MODELS_DIR, f"lgbm_fold_{i+1}.pkl")
        joblib.dump(model, path)
        print(f"    {path}")

    for i, model in enumerate(xgbm_models):
        path = os.path.join(MODELS_DIR, f"xgbm_fold_{i+1}.pkl")
        joblib.dump(model, path)
        print(f"    {path}")

    summary_path = os.path.join(MODELS_DIR, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    {summary_path}")

    feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    joblib.dump(FEATURE_COLS, feat_path)
    print(f"    {feat_path}")


# ============================================================
# SECTION 8: GENERATE SUBMISSION
# ============================================================

def generate_submission(df_test, test_preds_lgbm, test_preds_xgbm,
                        results, primary_model):
    print("\n[8/8] Generating submission")

    threshold = results[primary_model]["best_threshold"]
    print(f"    Model: {primary_model}  Threshold: {threshold:.3f}")

    if primary_model == "LightGBM":
        test_probs = test_preds_lgbm
    elif primary_model == "XGBoost":
        test_probs = test_preds_xgbm
    else:
        w_lgbm = results["Ensemble"]["weight_lgbm"]
        w_xgbm = results["Ensemble"]["weight_xgbm"]
        test_probs = (w_lgbm * test_preds_lgbm) + (w_xgbm * test_preds_xgbm)

    test_labels = (test_probs >= threshold).astype(int)

    pred_pos_rate = test_labels.mean() * 100
    print(f"    Predicted positive rate : {pred_pos_rate:.3f}%")
    print(f"    Training positive rate  : 0.856%")
    print(f"    Predicted class 1 count : {test_labels.sum():,} / {len(test_labels):,}")

    submission = pd.DataFrame({
        "ID"    : df_test["ID"].values,
        "target": test_labels,
    })
    submission.to_csv(SUBMIT_PATH, index=False)
    print(f"    Saved: {SUBMIT_PATH}  ({submission.shape[0]:,} rows)")
    return submission


# ============================================================
# SECTION 9: MAIN ORCHESTRATION
# ============================================================

def main():
    global df_test_global

    total_start = time.time()
    print("=" * 60)
    print("BINARY CLASSIFICATION PIPELINE")
    print("=" * 60)

    # 1. Load
    df_train, df_test = load_data()

    # 2. EDA
    run_eda(df_train, df_test)

    # 3. Feature engineering
    print("\n[3/8] Feature engineering...")
    df_train = engineer_features(df_train)
    df_test  = engineer_features(df_test)
    print(f"    Features: {len(FEATURE_COLS)} columns")
    print(f"    {FEATURE_COLS}")

    # Make test available globally for within-fold prediction
    df_test_global = df_test

    # 4. (Config defined in Section 0/4 at module level)
    print(f"\n[4/8] Model configuration")
    print(f"    LightGBM: {len(LGBM_PARAMS)} params  scale_pos_weight={SCALE_POS_WEIGHT}")
    print(f"    XGBoost : {len(XGBM_PARAMS)} params  scale_pos_weight={SCALE_POS_WEIGHT}")

    # 5. CV training
    (oof_lgbm, oof_xgbm,
     test_lgbm, test_xgbm,
     lgbm_models, xgbm_models, y_true) = train_with_cv(df_train)

    # 6. Evaluate and tune threshold
    results, primary = evaluate_and_tune_threshold(y_true, oof_lgbm, oof_xgbm)

    # 7. Save models
    save_models(lgbm_models, xgbm_models, results)

    # 8. Generate submission
    generate_submission(df_test, test_lgbm, test_xgbm, results, primary)

    total_min = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_min:.1f} minutes")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
