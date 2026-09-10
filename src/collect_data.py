import argparse
import csv
import os
import sys
import time

try:
    from src.hand_tracker import HandTracker
    from src.paths import RAW_DATASET_PATH
    from src.camera_utils import open_camera
except ImportError:
    from hand_tracker import HandTracker
    from paths import RAW_DATASET_PATH
    from camera_utils import open_camera

import cv2


def get_gesture_name(cli_label):

    if cli_label:
        name = cli_label.strip()
    else:
        name = input("Enter Gesture Name : ").strip()

    # Reject empty names and commas (would corrupt the CSV)
    while not name or "," in name:
        print("Invalid name. Gesture name cannot be empty or contain a comma.")
        name = input("Enter Gesture Name : ").strip()

    return name


def main():

    parser = argparse.ArgumentParser(description="Collect hand gesture landmark data.")
    parser.add_argument("--label", type=str, default=None, help="Name of the gesture to record.")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to collect.")
    args = parser.parse_args()

    gesture_name = get_gesture_name(args.label)
    target_samples = max(1, args.samples)

    tracker = HandTracker()

    print("Opening camera (this can take a couple of seconds)...")
    cap = open_camera()

    if cap is None:
        print("ERROR: Could not access the webcam. Check that it is connected, not in use "
              "by another app, and that desktop apps have camera permission in Windows Settings.")
        sys.exit(1)

    file_exists = os.path.exists(RAW_DATASET_PATH)

    sample_count = 0
    collecting = False
    last_save = time.time()

    print("\n==========================")
    print("Press S -> Start Recording")
    print("Press Q -> Quit")
    print(f"Target Samples -> {target_samples}")
    print("==========================\n")

    # Using 'with' guarantees the file is properly closed/flushed even on
    # a crash or an early exit (fixes silent data-loss risk).
    with open(RAW_DATASET_PATH, "a", newline="") as csv_file:

        writer = csv.writer(csv_file)

        if not file_exists:
            header = ["Gesture"]
            for i in range(21):
                header.extend([f"x{i}", f"y{i}", f"z{i}"])
            writer.writerow(header)

        while True:

            success, frame = cap.read()

            if not success:
                print("ERROR: Unable to read camera frame.")
                break
            frame, landmarks, bbox = tracker.find_hands(frame)
            cv2.putText(frame, f"Gesture : {gesture_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples : {sample_count}/{target_samples}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if collecting:
                cv2.putText(frame, "RECORDING...", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press S to Start", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Collect Gesture Data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                collecting = True
                print("Recording Started...")
            if key == ord("q"):
                break
            if collecting and len(landmarks) == 21:
                if time.time() - last_save > 0.2:
                    row = [gesture_name]
                    for point in landmarks:
                        row.extend([round(point[0], 6), round(point[1], 6), round(point[2], 6)])
                    writer.writerow(row)
                    sample_count += 1
                    print(f"Saved Sample : {sample_count}")
                    last_save = time.time()
                    if sample_count >= target_samples:
                        print(f"\n{target_samples} Samples Saved Successfully!")
                        break
    cap.release()
    cv2.destroyAllWindows()
    print("Dataset Saved Successfully!")
if __name__ == "__main__":
    main()
