import shutil
import sys
from collections import Counter, deque
from datetime import datetime

import cv2
import joblib
import numpy as np
import pickle
import pytesseract

try:
    from src.hand_tracker import HandTracker
    from src.paths import MODEL_PATH, ENCODER_PATH
    from src.camera_utils import open_camera
except ImportError:
    from hand_tracker import HandTracker
    from paths import MODEL_PATH, ENCODER_PATH
    from camera_utils import open_camera


# ============================================================
# TESSERACT OCR - auto-detect instead of a hardcoded Windows path
# ============================================================

def configure_tesseract():
    """
    Tries to find Tesseract automatically instead of relying on a
    hardcoded Windows path that breaks on every other machine.
    Returns True if Write Mode's OCR will work, False otherwise
    (Write Mode still runs, but word recognition is disabled).
    """

    # Already on PATH (Linux/Mac install, or Windows with PATH set)
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return True

    # Common Windows install locations, checked as a fallback
    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_windows_paths:
        try:
            pytesseract.pytesseract.tesseract_cmd = path
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            continue

    return False


# ============================================================
# GLOBAL STATE (Write Mode)
# ============================================================

write_mode = False
drawing = False
pen_color = (0, 0, 0)
pen_size = 6
ocr_available = False

canvas = np.ones((450, 1150, 3), dtype=np.uint8) * 255
sentence = []
canvas_history = []
previous_point = None

prediction_history = deque(maxlen=8)

WINDOW_NAME = "SignSpeak AI"

buttons = [
    ("ADD WORD", 20, 620, 150, 665, "add"),
    ("CLEAR", 165, 620, 285, 665, "clear"),
    ("NEW SENTENCE", 300, 620, 470, 665, "new"),
    ("UNDO", 485, 620, 590, 665, "undo"),
    ("RED", 605, 620, 680, 665, "red"),
    ("GREEN", 695, 620, 795, 665, "green"),
    ("BLUE", 810, 620, 890, 665, "blue"),
    ("YELLOW", 905, 620, 1005, 665, "yellow"),
    ("ERASER", 1020, 620, 1120, 665, "eraser"),
    ("CAMERA", 1135, 620, 1235, 665, "camera"),
]


# ============================================================
# CANVAS HELPERS
# ============================================================

def save_canvas_state():
    global canvas_history
    canvas_history.append(canvas.copy())
    if len(canvas_history) > 10:
        canvas_history.pop(0)


def clear_canvas():
    global canvas, canvas_history
    canvas_history.append(canvas.copy())
    canvas = np.ones((450, 1150, 3), dtype=np.uint8) * 255


def undo_canvas():
    global canvas, canvas_history
    if len(canvas_history) > 0:
        canvas = canvas_history.pop()


def new_sentence():
    global sentence
    sentence.clear()
    clear_canvas()


# ============================================================
# WORD RECOGNITION (shared by mouse button + keyboard shortcut)
# ============================================================

def add_word_to_sentence():
    """
    Single source of truth for "recognize what's on the canvas and
    append it to the sentence". Previously this logic was duplicated
    between the mouse ADD WORD button and the 'a' key handler.
    """

    if not ocr_available:
        print("Word recognition unavailable: Tesseract OCR was not found on this system.")
        return

    recognized_text = recognize_word()

    if recognized_text:
        sentence.append(recognized_text)
        print("Recognized:", recognized_text)
        clear_canvas()
    else:
        print("No word detected.")


def recognize_word():

    # Detect any pixel that differs meaningfully from pure white,
    # regardless of pen color. The old approach converted to grayscale
    # first and thresholded at 200 - but yellow's grayscale value
    # (~226) is so close to white (255) that it got wiped out along
    # with the background, making yellow writing invisible to OCR.
    # Comparing directly in color space fixes this for every pen color.
    white = np.full_like(canvas, 255)
    diff = cv2.absdiff(canvas, white)
    non_white_mask = (np.any(diff > 30, axis=2)).astype(np.uint8) * 255

    points = cv2.findNonZero(non_white_mask)
    if points is None:
        return ""

    x, y, w, h = cv2.boundingRect(points)

    padding = 30
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(non_white_mask.shape[1], x + w + padding)
    y2 = min(non_white_mask.shape[0], y + h + padding)

    # Build a clean black-strokes-on-white image for OCR, regardless of
    # what color the strokes actually were drawn in.
    cropped_mask = non_white_mask[y1:y2, x1:x2]
    word_image = np.where(cropped_mask > 0, 0, 255).astype(np.uint8)

    word_image = cv2.resize(word_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    word_image = cv2.GaussianBlur(word_image, (3, 3), 0)
    _, word_image = cv2.threshold(word_image, 180, 255, cv2.THRESH_BINARY)

    if len(points) < 15000 and h < 200:
        config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    else:
        config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    try:
        text = pytesseract.image_to_string(word_image, config=config)
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

    text = text.strip()

    cleaned_text = "".join(char for char in text if char.isalpha() or char == " ")
    text = " ".join(cleaned_text.split())

    return text


# ============================================================
# BUTTON HANDLING
# ============================================================

def draw_buttons(screen):

    for name, x1, y1, x2, y2, action in buttons:

        cv2.rectangle(screen, (x1, y1), (x2, y2), (55, 55, 55), -1)
        cv2.rectangle(screen, (x1, y1), (x2, y2), (150, 150, 150), 2)

        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        text_x = x1 + ((x2 - x1 - text_size[0]) // 2)
        text_y = y1 + ((y2 - y1 + text_size[1]) // 2)

        cv2.putText(screen, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)


def handle_button_click(x, y):

    global write_mode, pen_color, pen_size, sentence

    for name, x1, y1, x2, y2, action in buttons:

        if x1 <= x <= x2 and y1 <= y <= y2:

            if action == "add":
                add_word_to_sentence()
            elif action == "clear":
                clear_canvas()
            elif action == "new":
                new_sentence()
                print("New sentence started.")
            elif action == "undo":
                undo_canvas()
            elif action == "red":
                pen_color, pen_size = (0, 0, 255), 6
            elif action == "green":
                pen_color, pen_size = (0, 255, 0), 6
            elif action == "blue":
                pen_color, pen_size = (255, 0, 0), 6
            elif action == "yellow":
                pen_color, pen_size = (0, 255, 255), 6
            elif action == "eraser":
                pen_color, pen_size = (255, 255, 255), 25
            elif action == "camera":
                write_mode = False

            return


def mouse_callback(event, x, y, flags, param):

    global drawing, previous_point

    if event == cv2.EVENT_LBUTTONDOWN:

        if write_mode and 620 <= y <= 665:
            handle_button_click(x, y)
            return

        if not write_mode:
            return

        canvas_x = x - 65
        canvas_y = y - 100

        if not (0 <= canvas_x < canvas.shape[1] and 0 <= canvas_y < canvas.shape[0]):
            return

        save_canvas_state()
        drawing = True
        previous_point = (canvas_x, canvas_y)
        cv2.circle(canvas, previous_point, pen_size, pen_color, -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        canvas_x = x - 65
        canvas_y = y - 100

        if not (0 <= canvas_x < canvas.shape[1] and 0 <= canvas_y < canvas.shape[0]):
            return

        current_point = (canvas_x, canvas_y)
        cv2.line(canvas, previous_point, current_point, pen_color, pen_size)
        previous_point = current_point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        previous_point = None


# ============================================================
# WRITE SCREEN
# ============================================================

def create_write_screen():

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[:] = (25, 25, 25)

    cv2.rectangle(screen, (0, 0), (1280, 75), (35, 35, 35), -1)
    cv2.putText(screen, "SIGNSPEAK AI - WRITE MODE", (25, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(screen, "Write inside the white area", (925, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if not ocr_available:
        cv2.putText(screen, "(Word recognition disabled: Tesseract not found)", (925, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    cv2.rectangle(screen, (55, 90), (1225, 550), (0, 255, 0), 3)
    screen[100:550, 65:1215] = canvas

    cv2.rectangle(screen, (25, 560), (1255, 610), (0, 0, 0), -1)
    cv2.putText(screen, "Sentence:", (40, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    sentence_text = " ".join(sentence)
    if len(sentence_text) > 70:
        sentence_text = "..." + sentence_text[-67:]

    cv2.putText(screen, sentence_text, (170, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    draw_buttons(screen)

    cv2.putText(screen, "Click buttons with mouse  |  Keyboard shortcuts also work", (25, 705),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

    return screen


# ============================================================
# GESTURE PREDICTION
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


def predict_gesture(landmarks, model, encoder):
    """
    Returns a human-readable gesture label. Distinguishes between
    "no hand visible" and "hand visible but not confident enough" -
    previously both cases showed the same "No Hand" text, which was
    confusing since a bounding box would still be drawn on screen.
    """

    if len(landmarks) != 21:
        return "No Hand"

    features = np.array(landmarks).flatten()

    wrist_x, wrist_y, wrist_z = features[0], features[1], features[2]

    for i in range(0, 63, 3):
        features[i] -= wrist_x
        features[i + 1] -= wrist_y
        features[i + 2] -= wrist_z

    # Scale-normalize: without this, the same gesture produces different
    # feature magnitudes depending on how close the hand is to the camera,
    # which made the model much less confident (frequent "Uncertain...")
    # for hand distances that differ from training. Dividing by the
    # farthest landmark distance from the wrist makes the features
    # distance-invariant. This MUST match train_model.py exactly.
    distances = np.sqrt(features[0::3] ** 2 + features[1::3] ** 2 + features[2::3] ** 2)
    scale = np.max(distances)
    if scale > 0:
        features = features / scale

    features = features.reshape(1, -1)

    probabilities = model.predict_proba(features)
    confidence = np.max(probabilities)

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain..."

    prediction = model.predict(features)
    prediction_history.append(prediction[0])

    gesture_id = Counter(prediction_history).most_common(1)[0][0]
    gesture = encoder.inverse_transform([gesture_id])[0]

    return gesture


# ============================================================
# MAIN
# ============================================================

def main():

    global write_mode, ocr_available

    ocr_available = configure_tesseract()
    if not ocr_available:
        print("WARNING: Tesseract OCR not found. Write Mode will run, but word "
              "recognition (ADD WORD) will be disabled. Install Tesseract and "
              "make sure it's on your PATH to enable it.")

    try:
        model = joblib.load(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as file:
            encoder = pickle.load(file)
    except FileNotFoundError:
        print(f"ERROR: Model files not found at:\n  {MODEL_PATH}\n  {ENCODER_PATH}")
        print("Run collect_data.py, clean_dataset.py and train_model.py first.")
        sys.exit(1)

    print("AI Model Loaded Successfully!")

    tracker = HandTracker()

    print("Opening camera (this can take a couple of seconds)...")
    cap = open_camera()

    if cap is None:
        print("ERROR: Could not access the webcam. Check that it is connected, not in use "
              "by another app, and that desktop apps have camera permission in Windows Settings.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:

        # ========================================================
        # WRITE MODE
        # ========================================================
        if write_mode:

            write_screen = create_write_screen()
            cv2.imshow(WINDOW_NAME, write_screen)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                add_word_to_sentence()
            elif key == ord("c"):
                clear_canvas()
            elif key == ord("n"):
                new_sentence()
                print("New sentence started.")
            elif key == ord("u"):
                undo_canvas()
            elif key == ord("1"):
                globals()["pen_color"], globals()["pen_size"] = (0, 0, 255), 6
            elif key == ord("2"):
                globals()["pen_color"], globals()["pen_size"] = (0, 255, 0), 6
            elif key == ord("3"):
                globals()["pen_color"], globals()["pen_size"] = (255, 0, 0), 6
            elif key == ord("4"):
                globals()["pen_color"], globals()["pen_size"] = (0, 255, 255), 6
            elif key == ord("e"):
                globals()["pen_color"], globals()["pen_size"] = (255, 255, 255), 25
            elif key == ord("w"):
                write_mode = False
            elif key == ord("q"):
                break

            continue

        # ========================================================
        # CAMERA / GESTURE MODE
        # ========================================================
        success, frame = cap.read()

        if not success:
            print("Unable to read camera frame.")
            break

        frame, landmarks, bbox = tracker.find_hands(frame)

        gesture = predict_gesture(landmarks, model, encoder)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 3)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (35, 35, 35), -1)
        cv2.putText(frame, "SIGNSPEAK AI", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)

        cv2.circle(frame, (frame.shape[1] - 40, 35), 8, (0, 255, 0), -1)
        cv2.putText(frame, "LIVE", (frame.shape[1] - 90, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (10, 70), (370, 125), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture : {gesture}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (frame.shape[1] - 370, 70), (frame.shape[1] - 10, 115), (0, 0, 0), -1)
        current_time = datetime.now().strftime("%d-%m-%Y  %I:%M:%S %p")
        cv2.putText(frame, current_time, (frame.shape[1] - 360, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.rectangle(frame, (10, 140), (230, 190), (0, 0, 0), -1)
        cv2.putText(frame, "Press W : Write", (20, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(frame, "Press Q : Exit", (frame.shape[1] - 190, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            write_mode = True
            clear_canvas()
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
