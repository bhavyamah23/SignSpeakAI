import json

import numpy as np
import pandas as pd
import joblib
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    from src.paths import CLEAN_DATASET_PATH, MODEL_PATH, ENCODER_PATH, MODEL_INFO_PATH
except ImportError:
    from paths import CLEAN_DATASET_PATH, MODEL_PATH, ENCODER_PATH, MODEL_INFO_PATH


EXPECTED_FEATURE_COUNT = 63  # 21 landmarks x (x, y, z)
MIN_SAMPLES_PER_CLASS = 5
# A class needs at least this many samples for a stratified 80/20 split to
# put a meaningful number of samples in the test set.
MIN_SAMPLES_FOR_RELIABLE_TEST = 10


def main():

    print("Loading Dataset...")
    data = pd.read_csv(CLEAN_DATASET_PATH)
    print("Dataset Loaded Successfully!")
    print("Shape :", data.shape)

    # -----------------------------
    # Feature count sanity check
    # -----------------------------
    n_features = data.shape[1] - 1  # minus the "Gesture" column
    if n_features != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} landmark features but found "
            f"{n_features}. Check that collect_data.py / hand_tracker.py "
            f"still produce 21 landmarks with (x, y, z) each."
        )

    # -----------------------------
    # Remove classes with too few samples to train on at all
    # -----------------------------
    counts = data["Gesture"].value_counts()
    valid = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    data = data[data["Gesture"].isin(valid)]

    low_sample_classes = counts[(counts >= MIN_SAMPLES_PER_CLASS) & (counts < MIN_SAMPLES_FOR_RELIABLE_TEST)]
    if len(low_sample_classes) > 0:
        print("\nWARNING: These classes have few samples, so their test accuracy may be unreliable:")
        print(low_sample_classes)

    print("\nRemaining Classes")
    print(data["Gesture"].value_counts())

    # -----------------------------
    # Features
    # -----------------------------
    X = data.drop("Gesture", axis=1)
    y = data["Gesture"]

    # -----------------------------
    # Landmark Normalization (wrist-relative) - MUST match predictor.py
    # -----------------------------
    X = X.values.astype(float)

    for i in range(len(X)):
        wrist_x, wrist_y, wrist_z = X[i][0], X[i][1], X[i][2]
        for j in range(0, EXPECTED_FEATURE_COUNT, 3):
            X[i][j] -= wrist_x
            X[i][j + 1] -= wrist_y
            X[i][j + 2] -= wrist_z

        # Scale-normalize by the farthest landmark distance from the wrist,
        # so the model doesn't need to see every possible hand-to-camera
        # distance during training. MUST match predictor.py exactly.
        distances = np.sqrt(X[i][0::3] ** 2 + X[i][1::3] ** 2 + X[i][2::3] ** 2)
        scale = np.max(distances)
        if scale > 0:
            X[i] = X[i] / scale

    print("\nLandmarks Normalized!")

    # -----------------------------
    # Label Encoding
    # -----------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    print("\nClasses Found :")
    print(encoder.classes_)

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples :", len(X_test))

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    print("\nTraining AI Model...")

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    print("Training Completed!")

    # -----------------------------
    # Cross-validation (extra signal beyond the single train/test split)
    # -----------------------------
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        print("\n5-Fold Cross-Validation Accuracy : {:.2f}% (+/- {:.2f}%)".format(
            cv_scores.mean() * 100, cv_scores.std() * 100
        ))
    except ValueError as e:
        print(f"\nSkipped cross-validation (not enough samples per class): {e}")

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print("\nAccuracy :")
    print(round(accuracy * 100, 2), "%")

    print("\nClassification Report\n")
    print(classification_report(y_test, prediction, target_names=[str(x) for x in encoder.classes_]))

    # -----------------------------
    # Save Model + Metadata
    # -----------------------------
    joblib.dump(model, MODEL_PATH)

    with open(ENCODER_PATH, "wb") as file:
        pickle.dump(encoder, file)

    # Human-readable record of what this model expects, so a future
    # mismatch (e.g. landmark count changes) can be caught early instead
    # of silently producing garbage predictions.
    model_info = {
        "feature_count": EXPECTED_FEATURE_COUNT,
        "classes": list(encoder.classes_),
        "test_accuracy": round(float(accuracy) * 100, 2),
        "normalization": "wrist-relative translation + scale-normalized by max landmark distance from wrist",
    }
    with open(MODEL_INFO_PATH, "w") as file:
        json.dump(model_info, file, indent=2)

    print("\n==========================")
    print("MODEL SAVED SUCCESSFULLY")
    print("==========================")


if __name__ == "__main__":
    main()
