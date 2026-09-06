"""
SignSpeak AI - main entry point.

Run this from the project root:
    python app.py

You can also still run each script directly from inside src/ for
development (e.g. `cd src && python predictor.py`) - both ways work.
"""

from src.camera import start_camera
from src.predictor import main as run_full_app


def main():

    print("==========================================")
    print(" SignSpeak AI")
    print("==========================================")
    print("1. Full App (Gesture Recognition + Write Mode)")
    print("2. Quick Hand Detection Preview (no AI model needed)")
    print("==========================================")

    choice = input("Choose an option (1/2) : ").strip()

    if choice == "2":
        start_camera()
    else:
        run_full_app()


if __name__ == "__main__":
    main()
