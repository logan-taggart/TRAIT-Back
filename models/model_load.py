import os
import sys
from ultralytics import YOLO

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        full_path = os.path.join(base_path, "_internal", "models", relative_path)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_path, relative_path)

    return full_path

MODEL_PATH = resource_path("logo_detection.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)