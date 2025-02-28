import cv2
from flask import send_file
import io
from models.model_load import model
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils.embed import *

def compare_logo_embeddings(input_path, reference_path, model, feature_extractor, similarity_threshold=0.4):
    if feature_extractor == "BEiT":
        feature_extractor = BEiTEmbedding()
    elif feature_extractor == "CLIP":
        feature_extractor = CLIPEmbedding()
    else:
        print("Invalid feature extractor.")
        return

    input_logos, input_bboxes, input_img = extract_logo_regions(input_path, model)
    reference_logos, reference_bboxes, reference_img = extract_logo_regions(reference_path, model)
    
    if not input_logos or not reference_logos:
        print("No logos detected in one or both images.")
        return

    input_embeddings = [feature_extractor.extract_embedding(Image.fromarray(input_logo)) for input_logo in input_logos]
    reference_embeddings = [feature_extractor.extract_embedding(Image.fromarray(reference_logo)) for reference_logo in reference_logos]
    
    for index, input_embedding in enumerate(input_embeddings):
        for ref_index, reference_embedding in enumerate(reference_embeddings):
            similarity = compute_cosine_similarity(input_embedding, reference_embedding)

            print(f'similarity score: {similarity}')
            if similarity >= similarity_threshold:
                x1, y1, x2, y2 = input_bboxes[index]
                color = [255, 255, 255]
                cv2.rectangle(input_img, (x1, y1), (x2, y2), color, 2)

        _, img_encoded = cv2.imencode(".jpg", input_img)

    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/jpeg")

def compute_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)

    return cosine_similarity(embedding1, embedding2)


def extract_logo_regions(image, model):
    file_bytes = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model(img)

    logo_regions = []
    bounding_boxes = []

    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cropped_logo = img[y1:y2, x1:x2]

        if cropped_logo.size > 0:
            height, width = cropped_logo.shape[:2]

            new_height = 128
            new_width = int((new_height / height) * width)
            
            resized_logo = cv2.resize(cropped_logo, (new_width, new_height))

            logo_regions.append(resized_logo)
            bounding_boxes.append((x1, y1, x2, y2))

    return logo_regions, bounding_boxes, img


def identify_all_logos(file):
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