import cv2
import joblib
import pickle
import numpy as np
import time
from datetime import datetime
from collections import Counter, deque
from hand_tracker import HandTracker

# ==========================
# Load Model
# ==========================
model = joblib.load("../models/gesture_model.pkl")

with open("../models/label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

print("AI Model Loaded Successfully!")

# ==========================
# Initialize
# ==========================
tracker = HandTracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("SignSpeak AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "SignSpeak AI",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
prediction_history = deque(maxlen=20)

# ==========================
# Main Loop
# ==========================
while True:

    success, frame = cap.read()

    if not success:
        break

    frame, landmarks, bbox = tracker.find_hands(frame)

    gesture = "No Hand"

    if len(landmarks) == 21:

        features = np.array(landmarks).flatten()

        # Normalize using wrist
        wrist_x = features[0]
        wrist_y = features[1]
        wrist_z = features[2]

        for i in range(0, 63, 3):
            features[i] -= wrist_x
            features[i + 1] -= wrist_y
            features[i + 2] -= wrist_z

        features = features.reshape(1, -1)

        probabilities = model.predict_proba(features)
        confidence = np.max(probabilities)

        prediction = model.predict(features)

        prediction_history.append(prediction[0])

        gesture_id = Counter(prediction_history).most_common(1)[0][0]

        gesture = encoder.inverse_transform([gesture_id])[0]

    # -------------------------
# Hand Bounding Box
# -------------------------

    if bbox is not None:

       x1, y1, x2, y2 = bbox

       cv2.rectangle(
        frame,
        (x1 - 20, y1 - 20),
        (x2 + 20, y2 + 20),
        (0, 255, 0),
        3
    )

    # ==========================
# Header
# ==========================

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (35, 35, 35), -1)

    cv2.putText(
    frame,
    "SIGNSPEAK AI",
    (20, 45),
    cv2.FONT_HERSHEY_DUPLEX,
    1.1,
    (255, 255, 255),
    2
)

    cv2.circle(frame, (frame.shape[1]-40, 35), 8, (0,255,0), -1)

    cv2.putText(
            frame,
            "LIVE",
            (frame.shape[1]-90, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
)
    cv2.rectangle(
    frame,
    (10, 70),      # Top Left
    (370, 125),    # Bottom Right
    (0, 0, 0),     # Black Color
    -1             # Filled Rectangle
)

    cv2.putText(
        frame,
        f"Gesture : {gesture}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
    frame,
    "Press Q : Exit",
    (frame.shape[1]-190, frame.shape[0]-20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)
    cv2.rectangle(
    frame,
    (frame.shape[1]-370, 70),
    (frame.shape[1]-10, 115),
    (0, 0, 0),
    -1
)

    current_time = datetime.now().strftime("%d-%m-%Y  %I:%M:%S %p")

    cv2.putText(
        frame,
        current_time,
        (frame.shape[1]-360, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    cv2.imshow("SignSpeak AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()