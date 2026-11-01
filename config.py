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

APP_PAGE_TITLE = "Steel plate fault classifier"
APP_TITLE = "Steel plate fault classification"
APP_DESCRIPTION = (
    "Explore the dataset and compare five trained classifiers using your labeled "
    "test data. Models are loaded from saved artifacts; uploaded data is never "
    "used for training."
)
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]