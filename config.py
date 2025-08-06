# Healthcare Queue Optimizer Configuration

# Model paths
MODEL_DIR = "model"
MODEL_PATH = "model/healthcare_model.pt"
PROCESSOR_PATH = "model/data_processor.pkl"
METADATA_PATH = "model/training_metadata.json"

# Data paths
DATA_DIR = "data"
SYNTHETIC_DATA_PATH = "data/synthetic_data.csv"

# Model hyperparameters
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 32
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
TRAIN_SPLIT = 0.8

# Data validation ranges
AGE_RANGE = (0, 150)
HEART_RATE_RANGE = (30, 200)
SYSTOLIC_BP_RANGE = (70, 250)
DIASTOLIC_BP_RANGE = (40, 150)
DIABETES_VALUES = [0, 1]

# UI settings
PAGE_TITLE = "Healthcare Queue Optimizer"
PAGE_ICON = "🏥"
LAYOUT = "wide"

# Urgency colors for UI
URGENCY_COLORS = {
    "high": "background-color: #ffcccc; color: red;",
    "medium": "background-color: #fff3cd; color: #856404;",
    "low": "background-color: #d4edda; color: green;",
}

# Required CSV columns
REQUIRED_COLUMNS = [
    "patient_id",
    "age", 
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "history_diabetes",
    "symptom_text"
]

# Training data columns (includes labels)
TRAINING_COLUMNS = REQUIRED_COLUMNS + ["urgency_label"]
