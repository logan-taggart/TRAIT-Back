import cv2
from flask import send_file
import io
from models.model_load import model
import numpy as np

def process_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(img)

    confidence_threshold = 0.25
    color = (255, 255, 255)
    thickness = 2

    for box in results[0].boxes:
        if box.conf[0].item() > confidence_threshold:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    _, img_encoded = cv2.imencode(".jpg", img)

    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/jpeg")