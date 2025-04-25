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

def compare_logo_embeddings(input_file, reference_file, score_threshold, bb_color,bounding_box_threshold):
    thresholds = {
        'BEiTEmbedding': {'cosine': .3, 'euclidean': 110},
        'CLIPEmbedding': {'cosine': .65, 'euclidean': 7.5},
        'ResNetEmbedding': {'cosine': .75, 'euclidean': 50}
    }

    # Convert the input file to an image we can use with OpenCV and YOLO model
    img = convert_file_to_image(input_file)
    # Convert the reference file to an image we can use with OpenCV and YOLO model
    reference_img = convert_file_to_image(reference_file)

    # Convert the color from hex to BGR for OpenCV
    bb_color = hex_to_bgr(bb_color)

    # Get the bounding boxes for the logos in the input image and reference image
    # input_logos is the list of cropped logos
    # input_bboxes is the list of bounding boxes for each logo found in the image (tuple of (x1, y1, x2, y2))
    input_logos, input_bboxes, input_img = extract_logo_regions(img, bounding_box_threshold, return_img=True)
    # reference_logos is the list of cropped logos
    reference_logos, _ = extract_logo_regions(reference_img, bounding_box_threshold)

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
                
                # Add bounding box information to bounding_boxes_info
                bounding_boxes_info.append(extract_and_record_logo(input_img, input_bboxes[j], bb_color))

    return jsonify({
        "bounding_boxes": bounding_boxes_info,
        "image": img_to_base64(input_img)
    })
    


def identify_all_logos(file, bb_color, bounding_box_threshold):
    '''Returns the image with bounding boxes around all logos found'''
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # input_logos is the list of cropped logos
    # bounding_boxes is a list of bounding boxes for each logo found in the image (tuple of (x1, y1, x2, y2))
    input_logos, bounding_boxes = extract_logo_regions(img, bounding_box_threshold)

    # Convert for to bgr for OpenCV
    bb_color = hex_to_bgr(bb_color)
    
    # Use this to send bounding box info back to frontend
    bounding_boxes_info = []

    for box in bounding_boxes:
        
        # Save all of the bounding box info into this dict. This is sent to frontend
        bounding_boxes_info.append(extract_and_record_logo(img, box, bb_color))

    
    # Make image base64 so it can be jsonifyed.
    img_base64 = img_to_base64(img)

    return jsonify({
        "bounding_boxes": bounding_boxes_info,
        "image": img_base64
    })

