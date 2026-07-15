import cv2
from src.hand_tracker import HandTracker

tracker = HandTracker()

def start_camera():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame, landmarks, bbox = tracker.find_hands(frame)

        cv2.imshow("SignSpeak AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()