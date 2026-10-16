"""Very small Streamlit UI for the AI Data Scientist execution layer."""

from __future__ import annotations

from io import BytesIO

import joblib
import pandas as pd
import streamlit as st

from tools.dataset_profiler import profile_dataset
from tools.model_comparison import compare_models
from tools.model_training import train_candidates, train_final_model
from tools.predict import predict_rows
from tools.problem_detection import detect_problem_type
from tools.target_detection import detect_target


st.set_page_config(page_title="AI Data Scientist", page_icon="📊", layout="wide")
st.title("📊 AI Data Scientist")
st.caption("Upload a CSV, compare real validation results, choose a model, and make predictions.")


def reset_training() -> None:
    for key in ("training", "top_models", "comparison", "bundle", "bundle_bytes", "training_target"):
        st.session_state.pop(key, None)


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).rename(columns={"model_name": "Model"})


def confidence_flag(confidence: float) -> str:
    """Convert a classification confidence score into a simple user-facing flag."""
    if confidence >= 0.80:
        return "🟢 High confidence"
    if confidence >= 0.60:
        return "🟡 Moderate confidence"
    return "🔴 Low confidence"


def show_single_prediction(result: dict, target_column: str) -> None:
    """Present one prediction as a concise, non-technical result."""
    st.subheader("Prediction result")
    prediction = result["prediction"]
    probabilities = result.get("probabilities")

    if probabilities:
        confidence = float(probabilities.get(str(prediction), max(probabilities.values())))
        first, second = st.columns(2)
        first.metric(f"Predicted {target_column}", str(prediction))
        second.metric("Prediction confidence", f"{confidence:.1%}")
        st.info(f"{confidence_flag(confidence)} — based on the model's confidence in this classification.")
    else:
        st.metric(f"Predicted {target_column}", f"{float(prediction):,.4f}")


def prediction_table(predictions: list[dict], target_column: str) -> pd.DataFrame:
    """Flatten batch results into readable predicted values and confidence flags."""
    rows = []
    for prediction in predictions:
        row = {f"Predicted {target_column}": prediction["prediction"]}
        probabilities = prediction.get("probabilities", {})
        if probabilities:
            confidence = float(probabilities.get(str(prediction["prediction"]), max(probabilities.values())))
            row["Confidence flag"] = confidence_flag(confidence)
            row["Confidence"] = f"{confidence:.1%}"
        rows.append(row)
    return pd.DataFrame(rows)


uploaded = st.file_uploader("1. Upload a CSV dataset", type="csv")
if uploaded is None:
    st.info("Start by uploading a tabular CSV file.")
    st.stop()

file_marker = (uploaded.name, uploaded.size)
if st.session_state.get("file_marker") != file_marker:
    reset_training()
    st.session_state.file_marker = file_marker

try:
    df = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Could not read this CSV: {error}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The dataset needs at least two columns and one data row.")
    st.stop()

summary = profile_dataset(df)
detected = detect_target(df)
default_target = detected["target_column"] or df.columns[-1]
target_index = df.columns.get_loc(default_target)

left, right = st.columns([2, 1])
with left:
    st.subheader("Dataset")
    st.write(f"**{summary['rows']:,} rows** · **{summary['columns']} columns** · **{summary['duplicate_rows']:,} duplicate rows**")
    st.dataframe(df.head(), use_container_width=True)
with right:
    st.subheader("2. Select target")
    target = st.selectbox("Column to predict", options=df.columns.tolist(), index=target_index)
    if detected["target_column"]:
        st.caption(f"Auto-detection: {detected['reason']}")
    else:
        st.caption("Target was not confidently detected; select the column you want to predict.")
    try:
        problem = detect_problem_type(df, target)
        st.write(f"**Problem:** {problem['problem_type'].replace('_', ' ')}")
    except ValueError as error:
        st.error(str(error))
        st.stop()

with st.expander("Dataset profile"):
    st.json({
        "numeric_columns": summary["numeric_columns"],
        "categorical_columns": summary["categorical_columns"],
        "datetime_columns": summary["datetime_columns"],
        "missing_values": summary["missing_values"],
        "unique_values": summary["unique_values"],
    })

if st.button("3. Evaluate models", type="primary"):
    with st.spinner("Preprocessing data and training candidate models..."):
        try:
            training = train_candidates(df, target, problem["problem_type"])
            comparison, top_models = compare_models(training["results"], problem["problem_type"])
        except Exception as error:
            st.error(f"Training could not complete: {error}")
        else:
            st.session_state.training = training
            st.session_state.comparison = comparison
            st.session_state.top_models = top_models
            st.session_state.training_target = target
            st.session_state.problem_type = problem["problem_type"]
            st.session_state.pop("bundle", None)
            st.session_state.pop("bundle_bytes", None)

if st.session_state.get("training_target") != target:
    reset_training()

if "comparison" not in st.session_state:
    st.stop()

st.subheader("Validation results")
st.caption("These metrics are calculated on a held-out test split. They are not generated by an LLM.")
st.dataframe(metrics_table(st.session_state.comparison.to_dict(orient="records")), use_container_width=True)

st.subheader("4. Choose one of the two recommended models")
top_models = st.session_state.top_models
model_names = [model["model_name"] for model in top_models]
selected_name = st.radio("Recommended models", model_names, horizontal=True)
if st.button("Train selected model"):
    with st.spinner(f"Training {selected_name} on all labeled rows..."):
        bundle = train_final_model(st.session_state.training, selected_name)
        artifact = BytesIO()
        joblib.dump(bundle, artifact)
        st.session_state.bundle = bundle
        st.session_state.bundle_bytes = artifact.getvalue()
    st.success(f"{selected_name} trained successfully.")

if "bundle" not in st.session_state:
    st.stop()

st.download_button(
    "Download trained model (.joblib)",
    data=st.session_state.bundle_bytes,
    file_name="trained_model.joblib",
    mime="application/octet-stream",
)

st.subheader("5. Predict")
prediction_csv = st.file_uploader("Upload new rows CSV", type="csv", key="prediction_csv")
if prediction_csv is not None:
    try:
        new_rows = pd.read_csv(prediction_csv)
        if st.button("Predict uploaded rows"):
            predictions = predict_rows(st.session_state.bundle, new_rows)
            st.success(f"Predictions completed for {len(predictions):,} row(s).")
            st.dataframe(
                prediction_table(predictions, target), hide_index=True, use_container_width=True
            )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.caption("Or edit one example row and predict it.")
example = df.drop(columns=[target]).head(1)
row = st.data_editor(example, num_rows="fixed", use_container_width=True, key="single_row")
if st.button("Predict this row"):
    try:
        result = predict_rows(st.session_state.bundle, row)
        show_single_prediction(result[0], target)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
