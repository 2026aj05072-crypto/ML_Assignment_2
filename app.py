from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    APP_DESCRIPTION,
    APP_PAGE_TITLE,
    APP_TITLE,
    METRIC_COLUMNS,
    MODEL_DIR,
    TEST_DATA_PATH,
)
from metrics_utils import calculate_metrics


@st.cache_resource
def load_model(model_path: str):
    return load(model_path)


def main() -> None:
    st.set_page_config(
        page_title=APP_PAGE_TITLE,
        page_icon="🔩",
        layout="wide",
    )
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

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

    estimators = {
        name: load_model(str(MODEL_DIR / filename))
        for name, filename in model_files.items()
    }

    with st.sidebar:
        st.header("Evaluate models")
        uploaded_file = st.file_uploader(
            "Upload labeled test data (CSV)",
            type="csv",
            help="Upload test_data.csv or another CSV with all 27 features and FaultType.",
        )
        st.caption(
            "The repository's test_data.csv is ready to use. Custom files must "
            "include all required features and the FaultType target."
        )
        if TEST_DATA_PATH.is_file():
            st.download_button(
                "Download sample test CSV",
                data=TEST_DATA_PATH.read_bytes(),
                file_name=TEST_DATA_PATH.name,
                mime="text/csv",
            )
        else:
            st.warning("The sample test_data.csv file is not available.")
        selected_model = st.selectbox("Inspect a model", list(estimators))

    test_data = None
    test_features = None
    test_target = None
    if uploaded_file is not None:
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

    uploaded_classes = set(test_target.unique()) if test_target is not None else set()
    for model_name, estimator in estimators.items():
        unknown_classes = uploaded_classes.difference(estimator.classes_)
        if unknown_classes:
            st.error(
                f"The uploaded target contains classes not used to train "
                f"{model_name}: "
                f"{', '.join(map(str, sorted(unknown_classes, key=str)))}."
            )
            st.stop()

    overview_tab, metrics_tab, details_tab = st.tabs(
        ["Dataset overview", "Metrics comparison", "Model details"]
    )

    with overview_tab:
        st.subheader("Steel plates faults dataset")
        st.markdown(
            "Source: [UCI Machine Learning Repository]"
            "(https://archive.ics.uci.edu/dataset/198/steel+plates+faults)"
        )
        dataset_columns = st.columns(3)
        dataset_columns[0].metric("Instances", f"{metadata['dataset_rows']:,}")
        dataset_columns[1].metric(
            "Input features", f"{len(metadata['feature_columns']):,}"
        )
        dataset_columns[2].metric(
            "Fault classes", f"{len(metadata['target_classes']):,}"
        )

        class_balance = pd.DataFrame(metadata["class_distribution"]).rename(
            columns={"class": "Fault class", "count": "Instances"}
        )
        class_balance["Share"] = (
            class_balance["Instances"] / metadata["dataset_rows"] * 100
        )
        st.dataframe(
            class_balance,
            column_config={
                "Share": st.column_config.NumberColumn("Share", format="%.1f%%"),
            },
            hide_index=True,
            alt="Class balance in the full Steel Plates Faults dataset",
        )

        if test_target is not None:
            st.subheader("Uploaded test-set class balance")
            test_balance = (
                test_target.value_counts()
                .rename_axis("Fault class")
                .reset_index(name="Instances")
            )
            st.dataframe(
                test_balance,
                hide_index=True,
                alt="Class balance in the uploaded test data",
            )
        else:
            st.info("Upload a labeled test CSV from the sidebar to evaluate models.")

    if test_features is None or test_target is None:
        metrics_by_model = None
    else:
        metrics_by_model = {
            name: calculate_metrics(estimator, test_features, test_target)
            for name, estimator in estimators.items()
        }

    with metrics_tab:
        if metrics_by_model is None:
            st.info("Upload a labeled test CSV to compare model metrics.")
        else:
            results = pd.DataFrame.from_dict(metrics_by_model, orient="index")
            st.subheader("Test-set metrics")
            st.caption(
                "Metrics use the classes present in the uploaded test data; "
                "AUC is evaluated across those classes."
            )
            st.dataframe(
                results[METRIC_COLUMNS].round(4),
                alt="Evaluation metrics for all five classifiers",
            )
            selected_metric = st.selectbox(
                "Metric to compare",
                METRIC_COLUMNS,
                index=METRIC_COLUMNS.index("F1"),
            )
            chart_labels = {
                "Logistic Regression": "Logistic Reg.",
                "Decision Tree": "Decision Tree",
                "K-Nearest Neighbors": "KNN",
                "Gaussian Naive Bayes": "Gaussian NB",
                "Support Vector Machine": "SVM",
            }
            chart_data = results[[selected_metric]].reset_index(names="Model")
            chart_data["Model"] = chart_data["Model"].map(chart_labels)
            st.bar_chart(
                chart_data,
                x="Model",
                y=selected_metric,
                horizontal=True,
                height=400,
                alt=f"{selected_metric} comparison across all five classifiers",
            )

    with details_tab:
        if test_features is None or test_target is None:
            st.info("Upload a labeled test CSV to inspect a model.")
        else:
            estimator = estimators[selected_model]
            predictions = estimator.predict(test_features)
            classes = estimator.classes_

            st.subheader(f"{selected_model}: confusion matrix")
            st.caption("Rows show actual classes; columns show predicted classes.")
            matrix = confusion_matrix(test_target, predictions, labels=classes)
            labels = [str(label) for label in classes]
            matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
            matrix_frame.index.name = "Actual"
            matrix_frame.columns.name = "Predicted"
            st.dataframe(
                matrix_frame.style.background_gradient(cmap="Blues", axis=None),
                alt=f"Confusion matrix for {selected_model}",
            )

            st.subheader(f"{selected_model}: classification report")
            report = classification_report(
                test_target, predictions, output_dict=True, zero_division=0
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                alt=f"Classification report for {selected_model}",
            )


if __name__ == "__main__":
    main()
