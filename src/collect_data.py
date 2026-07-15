import cv2
import csv
import os
import time
from hand_tracker import HandTracker

# ==========================
# Initialize
# ==========================
tracker = HandTracker()

cap = cv2.VideoCapture(0)

gesture_name = input("Enter Gesture Name : ")

csv_path = "../dataset/gestures.csv"

file_exists = os.path.exists(csv_path)

csv_file = open(csv_path, "a", newline="")

writer = csv.writer(csv_file)

# ==========================
# Create Header
# ==========================
if not file_exists:

    header = ["Gesture"]

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    writer.writerow(header)

sample_count = 0
collecting = False
last_save = time.time()

print("\n==========================")
print("Press S -> Start Recording")
print("Press Q -> Quit")
print("==========================\n")

# ==========================
# Main Loop
# ==========================
while True:

    success, frame = cap.read()

    if not success:
        break

    frame, landmarks, bbox = tracker.find_hands(frame)

    cv2.putText(
        frame,
        f"Gesture : {gesture_name}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Samples : {sample_count}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    if collecting:
        cv2.putText(
            frame,
            "RECORDING...",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )
    else:
        cv2.putText(
            frame,
            "Press S to Start",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        collecting = True
        print("Recording Started...")

    if key == ord("q"):
        break

    # Save every 0.2 second
    if collecting and len(landmarks) == 21:

        if time.time() - last_save > 0.2:

            row = [gesture_name]

            for point in landmarks:

                row.extend([
                    round(point[0],6),
                    round(point[1],6),
                    round(point[2],6)
                ])

            writer.writerow(row)

            sample_count += 1

            print(f"Saved Sample : {sample_count}")

            last_save = time.time()

            # Stop after 50 samples
            if sample_count >= 50:

                print("\n50Samples Saved Successfully!")

                break

csv_file.close()

cap.release()

cv2.destroyAllWindows()

print("Dataset Saved Successfully!")