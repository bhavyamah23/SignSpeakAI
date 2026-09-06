import sys

import cv2

try:
    from src.hand_tracker import HandTracker
    from src.camera_utils import open_camera
except ImportError:
    from hand_tracker import HandTracker
    from camera_utils import open_camera


def start_camera():
    """
    Lightweight preview mode: shows the live camera feed with hand
    landmarks drawn on it, but does NOT run gesture prediction or
    Write Mode. Useful for quickly checking your camera / hand
    tracking is working before running the full app (predictor.py).
    For the full experience (gesture recognition + Write Mode), use
    predictor.main() instead - see app.py.
    """

    tracker = HandTracker()

    print("Opening camera (this can take a couple of seconds)...")
    cap = open_camera()

    if cap is None:
        print("ERROR: Could not access the webcam. Check that it is connected, not in use "
              "by another app, and that desktop apps have camera permission in Windows Settings.")
        sys.exit(1)

    print("Hand Detection Preview running. Press Q to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Unable to read camera frame.")
            break

        frame, landmarks, bbox = tracker.find_hands(frame)

        cv2.imshow("SignSpeak AI - Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
