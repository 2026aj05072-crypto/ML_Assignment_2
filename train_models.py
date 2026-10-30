from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from metrics_utils import build_preprocessor, calculate_metrics, create_classifiers


STEEL_FAULT_CLASSES = (
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the five assignment classifiers."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Dataset/Faults.NNA"),
        help="Dataset file (defaults to Dataset/Faults.NNA).",
    )
    parser.add_argument(
        "--target",
        help="Target column for a CSV dataset; Faults.NNA uses its one-hot labels.",
    )
    parser.add_argument(
        "--columns",
        type=Path,
        default=Path("Dataset/Faults27x7_var"),
        help="Column names for the whitespace-delimited Faults.NNA file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--test-output", type=Path, default=Path("test_data.csv"))
    return parser.parse_args()


def load_dataset(
    data_path: Path, target_column: str | None, columns_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    if data_path.suffix.lower() == ".nna":
        if target_column is not None:
            raise ValueError(
                "--target is not needed for Faults.NNA; its seven one-hot "
                "fault columns are converted into the target automatically."
            )
        if not columns_path.is_file():
            raise FileNotFoundError(f"Dataset column names not found: {columns_path}")

        columns = [
            line.strip()
            for line in columns_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate column names found in {columns_path}.")

        dataset = pd.read_csv(data_path, sep=r"\s+", header=None, names=columns)
        if dataset.shape[1] != len(columns):
            raise ValueError(
                f"Expected {len(columns)} columns from {columns_path}, "
                f"but found {dataset.shape[1]}."
            )
        if tuple(columns[-len(STEEL_FAULT_CLASSES) :]) != STEEL_FAULT_CLASSES:
            raise ValueError(
                "The final seven columns in the .var file must be the expected "
                "Steel Plates Faults labels."
            )

        label_data = dataset[list(STEEL_FAULT_CLASSES)]
        if not label_data.isin([0, 1]).all().all():
            raise ValueError("Fault label columns must contain only 0 or 1.")
        if not label_data.sum(axis=1).eq(1).all():
            raise ValueError(
                "Each Steel Plates Faults row must have exactly one active "
                "fault label."
            )

        target = label_data.idxmax(axis=1).rename("FaultType")
        features = dataset.drop(columns=list(STEEL_FAULT_CLASSES))
        return features, target

    if data_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a .NNA or .csv file.")
    if target_column is None:
        raise ValueError("--target is required when the dataset is a CSV file.")

    dataset = pd.read_csv(data_path)
    if target_column not in dataset.columns:
        raise ValueError(f"Target column {target_column!r} was not found in the CSV.")
    target = dataset[target_column]
    features = dataset.drop(columns=[target_column])
    return features, target


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")

    features, target = load_dataset(args.data, args.target, args.columns)
    if len(features) < 500:
        raise ValueError(
            f"The assignment requires at least 500 instances; found {len(features)}."
        )
    if target.isna().any():
        raise ValueError("The target column contains missing values.")

    if features.empty:
        raise ValueError("The dataset must contain at least one feature column.")
    if features.shape[1] < 12:
        raise ValueError(
            f"The assignment requires at least 12 input features; "
            f"found {features.shape[1]}."
        )
    if target.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )
    if train_target.value_counts().min() < 3:
        raise ValueError(
            "Each class needs at least three training rows for SVM calibration. "
            "Use more examples or a different test-size."
        )
    if test_target.nunique() != target.nunique():
        raise ValueError(
            "The test split must contain every target class to calculate AUC. "
            "Use a larger test-size."
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    test_data = test_features.copy()
    test_data[target.name] = test_target
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(args.test_output, index=False)

    metrics_by_model: dict[str, dict[str, float]] = {}
    model_files: dict[str, str] = {}

    for model_name, classifier in create_classifiers(args.random_state).items():
        estimator = Pipeline(
            [
                ("preprocessor", build_preprocessor(train_features)),
                ("classifier", classifier),
            ]
        )
        estimator.fit(train_features, train_target)

        model_slug = model_name.lower().replace(" ", "_").replace("-", "")
        model_path = args.model_dir / f"{model_slug}.joblib"
        dump(estimator, model_path)
        model_files[model_name] = model_path.name
        metrics_by_model[model_name] = calculate_metrics(
            estimator, test_features, test_target
        )

    metadata = {
        "dataset_name": (
            "Steel Plates Faults"
            if args.data.suffix.lower() == ".nna"
            else args.data.stem
        ),
        "target_column": target.name,
        "target_classes": sorted(target.unique().tolist()),
        "feature_columns": features.columns.tolist(),
        "model_files": model_files,
        "test_rows": len(test_data),
        "random_state": args.random_state,
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    results = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    results.to_csv(args.model_dir / "metrics.csv", index_label="Model")
    print(results.round(4).to_string())
    print(f"\nSaved models and metadata to: {args.model_dir}")
    print(f"Saved held-out test data to: {args.test_output}")


if __name__ == "__main__":
    main()
