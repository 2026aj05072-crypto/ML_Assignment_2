# Classification Model Comparison

Machine Learning Assignment 2 project comparing five classifiers on the Steel
Plates Faults dataset.

## a. Problem statement

Compare five classifiers on steel plate fault records and evaluate their
performance on a held-out test set. The Streamlit app evaluates the saved models
against a labeled CSV without retraining.

## b. Dataset description

- **Dataset:** [Steel Plates Faults (UCI)](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)
- **Instances:** 1,941
- **Input features:** 27 numeric features
- **Target:** `FaultType`, derived from the seven one-hot columns in the dataset
- **Classes:** Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults
- **Dataset files:** `Dataset/Faults.NNA` contains whitespace-delimited records;
  `Dataset/Faults27x7_var` contains the corresponding column names.
- **Preprocessing:** Each model pipeline imputes missing feature values, scales
  numeric inputs, and fits preprocessing on the training partition only.
  The seven one-hot fault columns are converted into one multiclass target.

The dataset exceeds the assignment minimums of 500 instances and 12 input
features.

## c. GitHub Repository Link
