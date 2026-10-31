from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "Dataset" / "Faults.NNA"
COLUMN_NAMES_PATH = PROJECT_DIR / "Dataset" / "Faults27x7_var"
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"

STEEL_FAULT_CLASSES = (
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
)

APP_PAGE_TITLE = "Classifier comparison"
APP_TITLE = "Classification Model Comparison"
APP_DESCRIPTION = (
    "Upload a labeled test CSV to evaluate the five saved classifiers. "
    "The app loads trained models and does not retrain on uploaded data."
)
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]