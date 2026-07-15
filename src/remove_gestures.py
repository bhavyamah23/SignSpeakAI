import pandas as pd

# Remove only Hello
remove = ["Hello"]

# Update gestures.csv
data = pd.read_csv("../dataset/gestures.csv")
data = data[~data["Gesture"].isin(remove)]
data.to_csv("../dataset/gestures.csv", index=False)

# Update gestures_clean.csv
data = pd.read_csv("../dataset/gestures_clean.csv")
data = data[~data["Gesture"].isin(remove)]
data.to_csv("../dataset/gestures_clean.csv", index=False)

print("✅ Hello removed successfully!")