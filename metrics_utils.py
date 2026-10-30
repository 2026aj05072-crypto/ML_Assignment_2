from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_classifiers(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(random_state=random_state), method="sigmoid", cv=3
        ),
    }


def calculate_metrics(
    estimator: Any, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = estimator.predict(features)
    classes = np.asarray(estimator.classes_)
    average = "binary" if len(classes) == 2 else "weighted"

    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(
            target,
            predictions,
            average=average,
            pos_label=classes[1] if len(classes) == 2 else None,
            zero_division=0,
        ),
        "Recall": recall_score(
            target,
            predictions,
            average=average,
            pos_label=classes[1] if len(classes) == 2 else None,
            zero_division=0,
        ),
        "F1": f1_score(
            target,
            predictions,
            average=average,
            pos_label=classes[1] if len(classes) == 2 else None,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(target, predictions),
    }

    probabilities = estimator.predict_proba(features)
    if len(classes) == 2:
        positive_target = (np.asarray(target) == classes[1]).astype(int)
        metrics["AUC"] = roc_auc_score(positive_target, probabilities[:, 1])
    else:
        metrics["AUC"] = roc_auc_score(
            target,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        )

    return {name: float(value) for name, value in metrics.items()}
