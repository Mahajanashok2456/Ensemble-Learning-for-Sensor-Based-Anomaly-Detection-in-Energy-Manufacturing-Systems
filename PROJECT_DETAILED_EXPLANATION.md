# Project Detailed Explanation Script

## 1) Project Title

**Binary Classification and Anomaly Detection Pipeline using XGBoost + LightGBM + Weighted Ensemble, deployed as a Streamlit Web Application**

---

## 2) Problem Statement

The goal is to detect rare positive cases (anomalies) in a **highly imbalanced binary classification** dataset.

- Target classes:
  - `0` = normal
  - `1` = anomaly
- Class imbalance is severe (~99.14% class 0), so normal accuracy is not enough.
- We need a robust model that gives high anomaly detection quality and can be used in a practical web app.

---

## 3) Tools, Libraries, and Technologies Used

### 3.1 Programming and App Layer

- **Python** for model training and inference
- **Streamlit** for web application UI

### 3.2 Data and Utilities

- **Pandas** for dataframe processing
- **NumPy** for numerical operations
- **PyArrow** for Parquet file support
- **Joblib** for saving/loading models and artifacts

### 3.3 Machine Learning Models

- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)

### 3.4 Validation and Metrics

- **Stratified K-Fold Cross-Validation** (5 folds)
- **scikit-learn metrics**: F1 score, Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix

---

## 4) Full-Form Glossary (Important Acronyms)

- **OOF**: Out-Of-Fold
- **CV**: Cross-Validation
- **EDA**: Exploratory Data Analysis
- **ADI**: Anomaly Detection Insights (in this project context)
- **AUC**: Area Under the Curve
- **ROC-AUC**: Receiver Operating Characteristic - Area Under the Curve
- **PR-AUC / AUC-PR**: Precision-Recall Area Under the Curve
- **F1**: Harmonic mean of Precision and Recall
- **TP / FP / TN / FN**: True Positive / False Positive / True Negative / False Negative
- **XGBoost**: Extreme Gradient Boosting
- **LightGBM**: Light Gradient Boosting Machine

---

## 5) End-to-End Steps (What We Did)

### Step 1: Data Loading

- Loaded `train.parquet` and `test.parquet`.
- Cast target to integer.
- Downcasted numeric columns to reduce memory usage.

### Step 2: Exploratory Data Analysis (EDA)

- Checked shape and null values.
- Analyzed class imbalance ratio.
- Inspected feature statistics and correlations.
- Performed temporal drift check by year.

### Step 3: Feature Engineering

Applied the same transformations to train and test:

1. **Date decomposition**: year, month, day, day_of_week, quarter, day_of_year, month-start/end, weekend
2. **Cyclical encoding**: `month_sin`, `month_cos`, `dow_sin`, `dow_cos`
3. **Decoded transformed source features**:
   - `X3_log = log(X3)`
   - `X4_log = log(X4)`
   - `X5_int = exp(X5)`
4. **Interaction features**:
   - `X1_minus_X2`
   - `X1_X2_ratio`
   - `X3log_X4log`

### Step 4: Model Training with Stratified CV

- Trained **XGBoost** and **LightGBM** in 5-fold stratified CV.
- Stored OOF probabilities and averaged test probabilities across folds.

### Step 5: Threshold Tuning

- Swept thresholds from 0.01 to 0.99.
- Chose threshold that maximized **F1 score**.
- Generated model-wise metrics and confusion matrix outputs.

### Step 6: Ensemble Creation

- Built weighted ensemble:

  $$p_{ensemble} = w_{lgbm} \cdot p_{lgbm} + w_{xgbm} \cdot p_{xgbm}$$

- Searched weights on grid (`0.00` to `1.00` for LightGBM weight).
- Tuned threshold for the ensemble as well.
- Selected best ensemble by highest F1.

### Step 7: Save Artifacts

Saved:

- `models/lgbm_fold_*.pkl`
- `models/xgbm_fold_*.pkl`
- `models/feature_cols.pkl`
- `models/eval_summary.json`

### Step 8: Submission + Web App

- Generated `submissions.csv`.
- Built Streamlit app for file upload prediction.
- Added anomaly counts, threshold visuals, distribution charts, and anomaly-only preview.

---

## 6) Metrics Chosen and Why

### Primary Optimization Metric

- **F1 score** at tuned threshold.
- Reason: In severe imbalance, F1 balances **Precision** and **Recall** for anomaly class.

### Ranking Metrics

- **PR-AUC (AUC-PR)**: more informative for rare positive class.
- **ROC-AUC**: overall separability metric.

### Operational Metrics

- **Precision**: anomaly correctness
- **Recall**: anomaly coverage
- **Confusion Matrix**: error type visibility (false alarms vs missed anomalies)

---

## 7) Current Model Results (from `models/eval_summary.json`)

### XGBoost Baseline

- ROC-AUC: **0.993827**
- PR-AUC: **0.856638**
- Best Threshold: **0.95**
- Best F1: **0.792685**
- Precision: **0.830679**
- Recall: **0.758014**

### LightGBM

- ROC-AUC: **0.994192**
- PR-AUC: **0.863820**
- Best Threshold: **0.95**
- Best F1: **0.800666**
- Precision: **0.832897**
- Recall: **0.770836**

### Ensemble (Final)

- ROC-AUC: **0.994348**
- PR-AUC: **0.865191**
- Best Threshold: **0.95**
- Best F1: **0.802934**
- Precision: **0.841042**
- Recall: **0.768129**
- Weights: `LightGBM = 0.9`, `XGBoost = 0.1`

### Interpretation

- Ensemble improves PR-AUC and F1 over both individual models.
- This indicates better balance in anomaly precision/recall while preserving strong ranking performance.

---

## 8) ADI (Anomaly Detection Insights) We Did

In this project, ADI is operationally represented by:

1. **Threshold-based anomaly flagging** (`prediction_probability >= threshold`)
2. **Anomaly statistics in app**:
   - total anomalies
   - anomaly rate
   - normal vs anomaly split
3. **Visual diagnostics**:
   - class distribution chart
   - threshold split chart
   - probability distribution
   - sorted probability vs threshold curve
4. **Anomaly-only preview table** (first 20 detected anomalies)

If your mentor intended **EDA** instead of ADI, we also completed full EDA in the pipeline before training.

---

## 9) Is the Final Model Fit? (Fit/Not Fit and Why)

### Fit Status: **Yes, fit for this problem context**

Reasoning:

- Strong CV metrics (very high ROC-AUC and PR-AUC)
- Improved ensemble F1 over individual models
- Threshold tuned for business-relevant anomaly detection behavior
- Robustness via Stratified K-Fold and OOF validation (reduces over-optimistic estimates)

### Practical Note

- “Fit” means suitable for this dataset and validation setup.
- Production fit should still include periodic monitoring, drift checks, and threshold recalibration.

---

## 10) Why We Used LightGBM and XGBoost

### LightGBM (Light Gradient Boosting Machine)

- Fast and efficient for large tabular data
- Strong handling of non-linear feature interactions
- Usually excellent with engineered tabular features

### XGBoost (Extreme Gradient Boosting)

- Very stable and well-regularized gradient boosting framework
- Strong generalization and proven performance in tabular ML

### Why Both

- They learn similar problem structures in different ways.
- Their errors are not perfectly identical.
- Combining them can improve overall robustness and anomaly detection quality.

---

## 11) Why We Combined Them (Ensemble Rationale)

- Ensemble reduces dependence on one model’s bias.
- Weighted blending improves stability and often improves minority-class performance.
- In your actual results, ensemble improved both **PR-AUC** and **F1**, which validates the design decision.

---

## 12) Why We Did Not Use Other Methods (Current Scope Decision)

### Not chosen in this phase

- Logistic Regression
- Support Vector Machine
- Deep Neural Networks
- Isolation Forest / One-Class methods
- CatBoost

### Why not now

1. The feature set and problem are tabular + imbalanced; gradient boosting is already highly competitive.
2. Existing pipeline constraints favored explainable, strong baseline-to-ensemble progression.
3. Training/inference simplicity and deployment speed were prioritized.
4. We already achieved high quality metrics with current approach.

This does not mean those methods are bad; it means they were outside current implementation scope.

---

## 13) Desired Outcome (What We Wanted)

- Build a **unique and practical anomaly detection solution**.
- Start with single models, then improve using ensemble.
- Convert model workflow into a usable web app for real file-based predictions.
- Provide clear anomaly counts, threshold-based insights, and downloadable results.

---

## 14) Final Architecture Summary

1. Data Load
2. EDA
3. Feature Engineering
4. Train XGBoost + LightGBM with Stratified CV
5. OOF evaluation + threshold tuning
6. Weighted Ensemble tuning
7. Save artifacts
8. Batch upload inference in Streamlit app with charts

---

## 15) Short Viva/Presentation Script

"This project solves imbalanced binary anomaly detection on tabular data. We first perform Exploratory Data Analysis and feature engineering, especially decoding transformed variables and creating date-cyclical and interaction features. We train two gradient boosting models: Extreme Gradient Boosting and Light Gradient Boosting Machine using Stratified K-Fold Cross-Validation. We evaluate using Out-Of-Fold predictions and optimize threshold for F1 score, while also tracking Precision-Recall Area Under Curve and Receiver Operating Characteristic Area Under Curve. Then we build a weighted ensemble and tune both weights and threshold. The final ensemble gives the best F1 and PR-AUC. We package this into a Streamlit web application for batch file upload, anomaly counts, threshold visualizations, and downloadable predictions."

---

## 16) Files in This Project (Core)

- `pipeline.py` -> end-to-end training, evaluation, ensemble, artifact saving, submission
- `app.py` -> Streamlit batch inference app
- `models/eval_summary.json` -> metrics + thresholds + ensemble weights
- `models/feature_cols.pkl` -> inference feature schema
- `models/lgbm_fold_*.pkl`, `models/xgbm_fold_*.pkl` -> trained fold models
- `submissions.csv` -> generated prediction output

---

## 17) Conclusion

The implemented approach follows a strong and practical sequence: **XGBoost baseline -> LightGBM -> weighted ensemble -> deployable web application**. Based on current CV metrics, the ensemble is the best-performing final model and is suitable for the project objective.
