import cv2
from hand_tracker import HandTracker

tracker = HandTracker()

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame, landmarks, bbox = tracker.find_hands(frame)

    if len(landmarks) != 0:
        print(landmarks)

    cv2.imshow("SignSpeakAI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()