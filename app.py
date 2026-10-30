from __future__ import annotations

import json

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import calculate_metrics


MODEL_DIR = Path(__file__).parent / "model"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_model(model_path: str):
    return load(model_path)


def main() -> None:
    st.set_page_config(page_title="Classifier comparison")
    st.title("Classification Model Comparison")
    st.write(
        "Upload a labeled test CSV to evaluate the five saved classifiers. "
        "The app loads trained models and does not retrain on uploaded data."
    )

    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.is_file():
        st.error("Model artifacts are missing. Run train_models.py first.")
        st.stop()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_files = metadata["model_files"]
    missing_models = [
        filename
        for filename in model_files.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing_models:
        st.error(f"Missing model files: {', '.join(missing_models)}")
        st.stop()

    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")
    if uploaded_file is None:
        st.info("Upload a CSV containing every feature and the target column.")
        st.stop()

    test_data = pd.read_csv(uploaded_file)
    target_column = metadata["target_column"]
    if target_column not in test_data.columns:
        st.error(f"The uploaded CSV must include target column {target_column!r}.")
        st.stop()

    feature_columns = metadata["feature_columns"]
    missing_features = [
        feature for feature in feature_columns if feature not in test_data.columns
    ]
    if missing_features:
        st.error(f"Missing required feature columns: {', '.join(missing_features)}")
        st.stop()
    if test_data[target_column].isna().any():
        st.error("The uploaded target column contains missing values.")
        st.stop()

    test_features = test_data[feature_columns]
    test_target = test_data[target_column]
    if test_target.nunique() < 2:
        st.error("The uploaded target column must contain at least two classes.")
        st.stop()

    estimators = {
        name: load_model(str(MODEL_DIR / filename))
        for name, filename in model_files.items()
    }
    uploaded_classes = set(test_target.unique())
    for model_name, estimator in estimators.items():
        if uploaded_classes != set(estimator.classes_):
            st.error(
                f"The uploaded target values must match the classes used to train "
                f"{model_name}."
            )
            st.stop()

    metrics_by_model = {
        name: calculate_metrics(estimator, test_features, test_target)
        for name, estimator in estimators.items()
    }

    st.subheader("Test-set metrics")
    results = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    st.dataframe(results[METRIC_COLUMNS].round(4), width="stretch")

    selected_model = st.selectbox("Inspect a model", list(estimators))
    estimator = estimators[selected_model]
    predictions = estimator.predict(test_features)
    classes = estimator.classes_

    st.subheader(f"{selected_model}: confusion matrix")
    matrix = confusion_matrix(test_target, predictions, labels=classes)
    labels = [str(label) for label in classes]
    st.dataframe(
        pd.DataFrame(matrix, index=labels, columns=labels),
        width="stretch",
    )

    st.subheader(f"{selected_model}: classification report")
    report = classification_report(
        test_target, predictions, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")


if __name__ == "__main__":
    main()
