import base64
import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from models.model_load import model


def hex_to_bgr(hex_color):
    '''Converts a hex color to bgr (blue, green, red)
    We do it this way because thats how OpenCV reads colors'''
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr


def compute_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.ravel(), embedding2.ravel())

def compute_euclidean_distances(embedding1, embedding2):
    return euclidean(embedding1.ravel(), embedding2.ravel())


def extract_logo_regions(image, save_crop=False, output_dir="cropped_logos", return_img=False):
    """Runs YOLO on an image and extracts detected logo regions."""
    # Check if input is a file path or an image array
    
    if isinstance(image, str):  # File path
        img = cv2.imread(image)
    else:  # Assume it's already an ndarray
        img = image

    if img is None:
        print("Error: Could not load image.")
        return [], [], None
    results = model(img)
    logo_regions = []
    bounding_boxes = []

    if save_crop and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, box in enumerate(results[0].boxes):
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cropped_logo = img[y1:y2, x1:x2]  # extract detected region

        if save_crop and cropped_logo.size > 0:
            cropped_logo_path = os.path.join(output_dir, f"cropped_logo_{idx}.jpg")
            cv2.imwrite(cropped_logo_path, cropped_logo)
            print(f"Logo {idx} saved: {cropped_logo_path}")

        if cropped_logo.size > 0:
            logo_regions.append(cropped_logo)
            bounding_boxes.append((x1, y1, x2, y2))
            print(f"Logo {idx} detected at coordinates: ({x1}, {y1}) -> ({x2}, {y2})")

    print(f"Total logos detected: {len(logo_regions)}")


    if return_img:
        return logo_regions, bounding_boxes, img
    else:
        return logo_regions, bounding_boxes


def img_to_base64(img):
    '''Converts an image (ndarray) to a base64-encoded string
    This is so we can send the image back to the frontend using a json object'''

    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')