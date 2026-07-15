import pandas as pd
import numpy as np
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading Dataset...")

data = pd.read_csv("../dataset/gestures_clean.csv")

print("Dataset Loaded Successfully!")

print("Shape :", data.shape)
# -----------------------------
# Remove Invalid Classes
# -----------------------------

counts = data["Gesture"].value_counts()

valid = counts[counts >= 5].index

data = data[data["Gesture"].isin(valid)]

print("\nRemaining Classes")

print(data["Gesture"].value_counts())
# -----------------------------
# Features
# -----------------------------

X = data.drop("Gesture", axis=1)

y = data["Gesture"]

# -----------------------------
# Landmark Normalization
# -----------------------------

X = X.values

for i in range(len(X)):

    wrist_x = X[i][0]
    wrist_y = X[i][1]
    wrist_z = X[i][2]

    for j in range(0,63,3):

        X[i][j] -= wrist_x
        X[i][j+1] -= wrist_y
        X[i][j+2] -= wrist_z

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
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))
# -----------------------------
# Train Random Forest
# -----------------------------

print("\nTraining AI Model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed!")
# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :")

print(round(accuracy * 100,2),"%")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    prediction,
    target_names=[str(x) for x in encoder.classes_]
))
# Save Model
# -----------------------------

joblib.dump(
    model,
    "../models/gesture_model.pkl"
)

with open("../models/label_encoder.pkl","wb") as file:

    pickle.dump(encoder,file)

print("\n==========================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================")