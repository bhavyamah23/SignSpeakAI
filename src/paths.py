"""
Central place for project paths.

Every script in src/ imports from here instead of using hardcoded
relative paths like "../dataset/...". This means the project works
correctly whether you run it as:
    python app.py                (from project root)
    cd src && python predictor.py (from inside src/)
"""

import os

# Project root = parent folder of src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_DATASET_PATH = os.path.join(DATASET_DIR, "gestures.csv")
CLEAN_DATASET_PATH = os.path.join(DATASET_DIR, "gestures_clean.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

# Make sure the folders exist so scripts never crash on a missing directory
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
