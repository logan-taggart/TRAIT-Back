import numpy as np
import cv2
from PIL import Image

from flask import jsonify


from utils.embed import BEiTEmbedding, CLIPEmbedding, ResNetEmbedding
# Trained YOLO model
from models.model_load import model

from utils.logo_detection_utils import *

beit = BEiTEmbedding()
clip = CLIPEmbedding()
resnet = ResNetEmbedding()

embedding_models = [beit, clip, resnet]

def compare_logo_embeddings(input_path, reference_path, score_threshold, bb_color,bounding_box_threshold):
    thresholds = {
        'BEiTEmbedding': {'cosine': .3, 'euclidean': 110},
        'CLIPEmbedding': {'cosine': .65, 'euclidean': 7.5},
        'ResNetEmbedding': {'cosine': .75, 'euclidean': 50}
    }

    bb_color = hex_to_bgr(bb_color)
    input_logos, input_bboxes, input_img = extract_logo_regions(input_path,bounding_box_threshold, return_img=True)
    reference_logos, _ = extract_logo_regions(reference_path,bounding_box_threshold)

    if not input_logos or not reference_logos:
        print("No logos detected in one or both images.")
        return jsonify({"error": "No logos detected in one or both images."}), 400

    # Initialize score tracker
    # Matrix of len(reference_logos) x len(input_logos). 
    # This keeps the scores separate for each reference logo found
    scores = [[0] * len(input_logos) for _ in range(len(reference_logos))]

    bounding_boxes_info = []  # Will contain bounding box data

    # For each embedding model
    for feature_extractor in embedding_models:
         # Get the name of the embedding model so we can index the thresholds dict
        model_name = type(feature_extractor).__name__

        # Get model-specific thresholds
        # This is from the dict defined at the function above
        cosine_threshold = thresholds[model_name]["cosine"]
        euclidean_threshold = thresholds[model_name]["euclidean"]

        # Compute embeddings and put them into an array
        input_embeddings = [feature_extractor.extract_embedding(Image.fromarray(logo)) for logo in input_logos]
        reference_embeddings = [feature_extractor.extract_embedding(Image.fromarray(logo)) for logo in reference_logos]


        # Iterate through each embeddings (basically each logo)
        for i, ref_embedding in enumerate(reference_embeddings):
            ref_embedding = ref_embedding.reshape(1, -1)  # Ensure 2D
            for j, input_embedding in enumerate(input_embeddings):

                input_embedding = input_embedding.reshape(1, -1)  # Ensure 2D
                # Compute similarity scores
                cosine_sim = compute_cosine_similarity(input_embedding, ref_embedding)
                euclidean_dist = compute_euclidean_distances(input_embedding, ref_embedding)

                # Check if similarities meet the model specific thresholds
                # Again, scores is a 2d array. Rows = num of reference images, cols = num of main image
                if cosine_sim >= cosine_threshold:
                    scores[i][j] += 1 # Plus 1 if cosine sim is met
                if euclidean_dist <= euclidean_threshold:
                    scores[i][j] += 1 # Plus 1 if euclidean distance is met

                # Print what the scores are
                print(f'{model_name} score: {scores[i][j]}')

    # Final decision: Classify as match if score is at least score_threshold/6
    for i in range(len(reference_logos)): # Iterate over reference logos (rows)
        for j in range(len(input_logos)): # Iterate over input logos (columns)
            
            if scores[i][j] >= score_threshold: # Check per reference logo
                x1, y1, x2, y2 = input_bboxes[j]

                cv2.rectangle(input_img, (x1, y1), (x2, y2), bb_color, 2)

                box_area = round((x2 - x1) * (y2 - y1), 2)
                box_height = round(y2 - y1, 2)
                box_width = x2 - x1
                image_height, image_width = input_img.shape[:2]
                total_image_area = image_width * image_height
                coverage_percentage = round((box_area / total_image_area) * 100, 2)
                # Add bounding box information to bounding_boxes_info
                bounding_boxes_info.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": scores[i][j],
                    "box_width": box_width,
                    "box_height":box_height,
                    "box_area": box_area,
                    "box_coverage_percentage": coverage_percentage,
                    "cropped_logo": img_to_base64(input_img[y1:y2, x1:x2])
                })

    return jsonify({
        "bounding_boxes": bounding_boxes_info,
        "image": img_to_base64(input_img)
    })
    


def identify_all_logos(file, bb_color, bounding_box_threshold):
    '''Returns the image with bounding boxes around all logos found'''
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
   
    results = model(img)

    confidence_threshold = bounding_box_threshold
    bb_color = hex_to_bgr(bb_color)
    thickness = 2

    # Get the image width and height
    image_height, image_width = img.shape[:2]
    # Get the total image area
    total_image_area = image_width * image_height

    # Use this to send bounding box info back to frontend
    bounding_boxes_info = []

    for box in results[0].boxes:
        if box.conf[0].item() > confidence_threshold:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # Calculate the bounding box area
            box_width = round(x2 - x1, 2)
            box_height = round(y2 - y1, 2)
            box_area = round(box_width * box_height, 2)

            # Calculate percentage coverage of the logo within the image
            coverage_percentage = round((box_area / total_image_area) * 100, 2)

            # Save all of the bounding box info into this dict. This is sent to frontend
            bounding_boxes_info.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": box_width,
                "box_height": box_height,
                "box_area": box_area,
                "box_coverage_percentage": coverage_percentage,
                "cropped_logo": img_to_base64(img[y1:y2, x1:x2])
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), bb_color, thickness)

    
    # Make image base64 so it can be jsonifyed.
    img_base64 = img_to_base64(img)

    return jsonify({
        "bounding_boxes": bounding_boxes_info,
        "image": img_base64
    })