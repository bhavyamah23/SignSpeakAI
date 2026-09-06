"""
Shared helper for opening the webcam reliably on Windows.

Some Windows setups report the camera as "opened" successfully but then
fail to deliver the very first frame(s) - the camera needs a brief
moment to warm up. This helper tries the DirectShow backend (more
reliable on Windows than the default backend) and retries reading a
frame a few times before giving up, instead of failing on the very
first read like the original code did.
"""

import time

import cv2


def open_camera(index=0, warmup_attempts=15, warmup_delay=0.2):

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None

    # Warm-up: try reading a few frames before handing the camera back.
    # This avoids the "camera opened but first read fails" issue seen
    # on some Windows machines/drivers.
    for _ in range(warmup_attempts):
        success, _ = cap.read()
        if success:
            return cap
        time.sleep(warmup_delay)

    # Camera opened but never produced a frame - treat as unusable.
    cap.release()
    return None
