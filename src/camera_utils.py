import time

import cv2


def open_camera(index=0, warmup_attempts=15, warmup_delay=0.2):

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None
    for _ in range(warmup_attempts):
        success, _ = cap.read()
        if success:
            return cap
        time.sleep(warmup_delay)
    cap.release()
    return None
