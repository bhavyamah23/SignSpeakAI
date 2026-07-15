import pandas as pd

# Load cleaned dataset
data = pd.read_csv("../dataset/gestures_clean.csv")

print("\n===== Gesture Count =====\n")
print(data["Gesture"].value_counts())

print("\n===== Total Samples =====")
print(len(data))