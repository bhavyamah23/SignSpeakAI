import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self):

        # MediaPipe Hands
        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # Drawing Utility
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        landmark_list = []

        bbox = None

        if results.multi_hand_landmarks:

            # Take only first hand
            hand = results.multi_hand_landmarks[0]

            h, w, c = frame.shape

            x_list = []
            y_list = []

            if draw:

                self.mpDraw.draw_landmarks(
                    frame,
                    hand,
                    self.mpHands.HAND_CONNECTIONS
                )

            for idx, lm in enumerate(hand.landmark):

                px = int(lm.x * w)
                py = int(lm.y * h)

                x_list.append(px)
                y_list.append(py)

                landmark_list.append([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            xmin = min(x_list)
            xmax = max(x_list)

            ymin = min(y_list)
            ymax = max(y_list)

            bbox = (xmin, ymin, xmax, ymax)

            if draw:

                cv2.rectangle(
                    frame,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Hand Detected",
                    (xmin - 20, ymin - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        return frame, landmark_list, bbox