from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EVAL_SUMMARY_PATH = MODELS_DIR / "eval_summary.json"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"


@st.cache_resource
def load_artifacts() -> dict[str, Any]:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Models directory not found at: {MODELS_DIR}. Run pipeline.py first."
        )

    if not EVAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing eval summary file: {EVAL_SUMMARY_PATH}. Run pipeline.py first."
        )

    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLS_PATH}. Run pipeline.py first."
        )

    with EVAL_SUMMARY_PATH.open("r", encoding="utf-8") as file_obj:
        eval_summary = json.load(file_obj)

    feature_cols = joblib.load(FEATURE_COLS_PATH)

    lgbm_paths = sorted(MODELS_DIR.glob("lgbm_fold_*.pkl"))
    xgbm_paths = sorted(MODELS_DIR.glob("xgbm_fold_*.pkl"))

    if not lgbm_paths and not xgbm_paths:
        raise FileNotFoundError(
            "No model files were found (lgbm_fold_*.pkl / xgbm_fold_*.pkl). Run pipeline.py first."
        )

    lgbm_models = [joblib.load(model_path) for model_path in lgbm_paths]
    xgbm_models = [joblib.load(model_path) for model_path in xgbm_paths]

    return {
        "eval_summary": eval_summary,
        "feature_cols": feature_cols,
        "lgbm_models": lgbm_models,
        "xgbm_models": xgbm_models,
    }


def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    output_df["Date"] = pd.to_datetime(output_df["Date"])

    output_df["year"] = output_df["Date"].dt.year.astype(np.int16)
    output_df["month"] = output_df["Date"].dt.month.astype(np.int8)
    output_df["day"] = output_df["Date"].dt.day.astype(np.int8)
    output_df["day_of_week"] = output_df["Date"].dt.dayofweek.astype(np.int8)
    output_df["quarter"] = output_df["Date"].dt.quarter.astype(np.int8)
    output_df["day_of_year"] = output_df["Date"].dt.dayofyear.astype(np.int16)
    output_df["is_month_end"] = output_df["Date"].dt.is_month_end.astype(np.int8)
    output_df["is_month_start"] = output_df["Date"].dt.is_month_start.astype(np.int8)
    output_df["is_weekend"] = (output_df["Date"].dt.dayofweek >= 5).astype(np.int8)

    output_df["month_sin"] = np.sin(2 * np.pi * output_df["month"] / 12).astype(np.float32)
    output_df["month_cos"] = np.cos(2 * np.pi * output_df["month"] / 12).astype(np.float32)
    output_df["dow_sin"] = np.sin(2 * np.pi * output_df["day_of_week"] / 7).astype(np.float32)
    output_df["dow_cos"] = np.cos(2 * np.pi * output_df["day_of_week"] / 7).astype(np.float32)

    output_df["X3_log"] = np.round(np.log(output_df["X3"].clip(lower=1e-10))).astype(np.int16)
    output_df["X4_log"] = np.round(np.log(output_df["X4"].clip(lower=1e-10))).astype(np.int16)
    output_df["X5_int"] = np.round(np.exp(output_df["X5"])).astype(np.int8)

    output_df["X1_minus_X2"] = (output_df["X1"] - output_df["X2"]).astype(np.float32)
    output_df["X1_X2_ratio"] = (output_df["X1"] / (output_df["X2"] + 1e-8)).astype(np.float32)
    output_df["X3log_X4log"] = (
        output_df["X3_log"].astype(np.int32) * output_df["X4_log"].astype(np.int32)
    ).astype(np.int32)

    return output_df


def average_probabilities(models: list[Any], features: pd.DataFrame) -> np.ndarray:
    if not models:
        return np.zeros(len(features), dtype=np.float64)

    probs = [model.predict_proba(features)[:, 1] for model in models]
    stacked = np.vstack(probs)
    return np.mean(stacked, axis=0)


def get_primary_model(eval_summary: dict[str, Any]) -> str:
    if "Ensemble" in eval_summary:
        return "Ensemble"
    return max(eval_summary, key=lambda model_name: eval_summary[model_name]["auc_pr"])


def resolve_model_name(model_choice: str, primary_model: str) -> str:
    if model_choice == "Auto (best AUC-PR)":
        return primary_model
    if model_choice not in {"Ensemble", "LightGBM", "XGBoost"}:
        return primary_model
    if model_choice == "LightGBM":
        return "LightGBM"
    if model_choice == "XGBoost":
        return "XGBoost"
    if model_choice == "Ensemble":
        return "Ensemble"
    return "XGBoost"


def read_uploaded_dataframe(uploaded_file: Any) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file format. Please upload a .csv or .parquet file.")


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = {"Date", "X1", "X2", "X3", "X4", "X5"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Uploaded file is missing required columns: {missing_str}")


def build_model_summary_table(eval_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in eval_summary.items():
        rows.append(
            {
                "Model": model_name,
                "AUC-ROC": metrics.get("auc_roc"),
                "AUC-PR": metrics.get("auc_pr"),
                "Best F1": metrics.get("best_f1"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "Threshold": metrics.get("best_threshold"),
            }
        )
    return pd.DataFrame(rows).sort_values(by="AUC-PR", ascending=False)


def build_probability_histogram(probabilities: np.ndarray, bins: int = 30) -> pd.DataFrame:
    hist_counts, bin_edges = np.histogram(probabilities, bins=bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return pd.DataFrame(
        {
            "Probability": np.round(bin_centers, 4),
            "Count": hist_counts,
        }
    )


def build_threshold_comparison(probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    below_or_equal = int((probabilities <= threshold).sum())
    above = int((probabilities > threshold).sum())
    return pd.DataFrame(
        {
            "Group": ["<= Threshold", "> Threshold"],
            "Count": [below_or_equal, above],
        }
    )


def main() -> None:
    st.set_page_config(page_title="Binary Classifier Frontend", page_icon="📊", layout="wide")
    st.title("📊 Binary Classification Frontend")
    st.caption("Predict anomalies from uploaded files using your trained fold models.")

    try:
        with st.spinner("Loading model artifacts..."):
            artifacts = load_artifacts()
    except Exception as exc:
        st.error(str(exc))
        return

    eval_summary = artifacts["eval_summary"]
    feature_cols = artifacts["feature_cols"]
    lgbm_models = artifacts["lgbm_models"]
    xgbm_models = artifacts["xgbm_models"]

    primary_model = get_primary_model(eval_summary)
    with st.sidebar:
        st.header("⚙️ Prediction Settings")
        model_options = ["Auto (best AUC-PR)"]
        if "Ensemble" in eval_summary:
            model_options.append("Ensemble")
        model_options.extend(["LightGBM", "XGBoost"])

        model_choice = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Auto selects the model with the highest AUC-PR from eval summary.",
        )
        selected_model = resolve_model_name(model_choice, primary_model)
        selected_threshold = float(eval_summary[selected_model]["best_threshold"])
        st.metric("Active Threshold", f"{selected_threshold:.4f}")
        st.caption(f"Primary model by AUC-PR: {primary_model}")

        st.divider()
        st.markdown("### 📝 Input Requirements")
        st.markdown("- Upload file: Date, X1, X2, X3, X4, X5")
        st.markdown("- File formats: .csv, .parquet")

    overview_col, table_col = st.columns([1, 2])
    with overview_col:
        st.info(
            "Upload a file to generate anomaly predictions. "
            "Predictions are generated with fold-model averaging."
        )
    with table_col:
        st.markdown("#### Model Performance Summary")
        st.dataframe(build_model_summary_table(eval_summary), use_container_width=True, hide_index=True)

    st.subheader("Predict from Uploaded File")
    st.caption("Upload a .csv or .parquet file with Date, X1, X2, X3, X4, X5 columns.")
    uploaded_file = st.file_uploader(
        "Choose file",
        type=["csv", "parquet"],
    )

    if uploaded_file is not None:
        try:
            batch_progress = st.progress(0, text="Starting batch prediction...")

            raw_df = read_uploaded_dataframe(uploaded_file)
            batch_progress.progress(15, text="File loaded. Validating schema...")
            validate_required_columns(raw_df)

            input_cols = ["Date", "X1", "X2", "X3", "X4", "X5"]
            input_df = raw_df[input_cols].copy()
            batch_progress.progress(35, text="Engineering features...")
            feature_df = engineer_features_for_inference(input_df)
            feature_df = feature_df[feature_cols]

            batch_progress.progress(55, text="Running model inference...")
            lgbm_probs = average_probabilities(lgbm_models, feature_df)
            xgbm_probs = average_probabilities(xgbm_models, feature_df)

            final_model_name = resolve_model_name(model_choice, primary_model)
            if final_model_name == "LightGBM":
                final_probs = lgbm_probs
            elif final_model_name == "XGBoost":
                final_probs = xgbm_probs
            else:
                ensemble_cfg = eval_summary.get("Ensemble", {})
                weight_lgbm = float(ensemble_cfg.get("weight_lgbm", 0.5))
                weight_xgbm = float(ensemble_cfg.get("weight_xgbm", 0.5))
                final_probs = (weight_lgbm * lgbm_probs) + (weight_xgbm * xgbm_probs)

            threshold = float(eval_summary[final_model_name]["best_threshold"])
            final_labels = (final_probs >= threshold).astype(int)

            batch_progress.progress(75, text="Preparing result tables and charts...")
            results_df = raw_df.copy()
            results_df["prediction_probability"] = final_probs
            results_df["prediction_label"] = final_labels
            results_df["model_used"] = final_model_name
            results_df["threshold_used"] = threshold

            anomaly_count = int(final_labels.sum())
            total_count = len(results_df)
            normal_count = total_count - anomaly_count
            anomaly_rate = (anomaly_count / total_count) * 100 if total_count > 0 else 0.0

            st.success(f"Predictions completed for {total_count:,} rows.")
            batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)
            batch_col1.metric("Rows", f"{total_count:,}")
            batch_col2.metric("Anomalies Detected", f"{anomaly_count:,}")
            batch_col3.metric("Normal Records", f"{normal_count:,}")
            batch_col4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

            st.markdown("#### Anomaly & Threshold Visuals")
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                class_distribution_df = pd.DataFrame(
                    {
                        "Label": ["Normal (0)", "Anomaly (1)"],
                        "Count": [normal_count, anomaly_count],
                    }
                )
                st.markdown("**Class Distribution**")
                st.bar_chart(class_distribution_df.set_index("Label"))

                threshold_df = build_threshold_comparison(final_probs, threshold)
                st.markdown(f"**Threshold Split (threshold = {threshold:.4f})**")
                st.bar_chart(threshold_df.set_index("Group"))

            with viz_col2:
                hist_df = build_probability_histogram(final_probs, bins=30)
                st.markdown("**Probability Distribution**")
                st.line_chart(hist_df.set_index("Probability")["Count"])

                sampled_probs = np.sort(final_probs)
                if len(sampled_probs) > 5000:
                    sample_idx = np.linspace(0, len(sampled_probs) - 1, 5000).astype(int)
                    sampled_probs = sampled_probs[sample_idx]
                trend_df = pd.DataFrame(
                    {
                        "Sorted Probability": sampled_probs,
                        "Threshold": np.full(len(sampled_probs), threshold),
                    }
                )
                st.markdown("**Sorted Probabilities vs Threshold**")
                st.line_chart(trend_df)

            st.markdown("#### Anomaly Preview (first 20 anomaly rows)")
            anomaly_preview_df = results_df[results_df["prediction_label"] == 1].head(20)
            if anomaly_preview_df.empty:
                st.info("No anomalies detected for the uploaded file at the current threshold.")
            else:
                st.dataframe(anomaly_preview_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            batch_progress.progress(95, text="Creating downloadable output...")
            st.download_button(
                label="Download predictions (CSV)",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            batch_progress.progress(100, text="Batch prediction complete.")
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
