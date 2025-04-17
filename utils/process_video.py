import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tr
import cv2
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BeitFeatureExtractor, BeitModel
from torchvision.models.feature_extraction import create_feature_extractor
from ultralytics import YOLO
import os
from collections import defaultdict
import numpy as np

from utils.embed import BEiTEmbedding, CLIPEmbedding, ResNetEmbedding
beit = BEiTEmbedding()
clip = CLIPEmbedding()
resnet = ResNetEmbedding()

embedding_models = [beit, clip, resnet]

import imageio
import io
import imageio.v3 as iio
from scipy.spatial.distance import cosine, euclidean
from flask import jsonify
import base64
from PIL import Image

from models.model_load import model


def extract_logo_regions(image, save_crop=False, output_dir="cropped_logos"):
    """Runs YOLO on an image and extracts detected logo regions."""
    # Check if input is a file path or an image array
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    if img is None:
        print("Error: Could not load image.")
        return [], []

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

    return logo_regions, bounding_boxes

def draw_bb_box(bbox, frame, ID):
    '''
    Draws the bounding box around the logo in the frame
    
    bbox is the coordinates of the bouning box
    frame is the image we want to draw on
    ID is the identification of the detected logo
    '''
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
    cv2.putText(frame, f"ID: {ID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (77, 33, 191), 2)



def process_video(input_video_path, frame_skip=5):
    # resnet size = 2048
    # clip size = 512
    # beit size = 768
    embedding_dim = 768 # Size for BEIT
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    logo_id_counter = 0 # How many unique logos we've seen
    logo_id_map = {}  # maps FAISS index to logo ID
    logo_appearance_counts = defaultdict(int) # How many times a unique logo has appeared
    
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_idx = 0
    save_frame = False
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # stop if video ends

        if frame_idx % frame_skip == 0:  # process every 5th frame
            print(f"Processing frame {frame_idx}")

            # extract detected logos from the current frame
            input_logos, input_bboxes = extract_logo_regions(frame, save_crop=False)

            # draw a bounding box around each detected logo
            for input_logo, bbox in zip(input_logos, input_bboxes):
                embedding = embedding_models[0].extract_embedding(Image.fromarray(input_logo))
                faiss.normalize_L2(embedding) # normalize the embedding. Works really well with FAISS

                if faiss_index.ntotal == 0: # First entry into FAISS
                    print("ADDING NEW INDEX") # Create a new index into FAISS
                    faiss_index.add(embedding) # Get embedding
                    logo_id_map[0] = logo_id_counter # First unique logo
                    logo_appearance_counts[logo_id_counter] += 1 # increment the unique logo
                    logo_id_counter += 1 # Go to next unique ID
                    assigned_id = logo_id_counter - 1 # current unique ID
                    save_frame = True
                else:
                    # Get the L2 distance and index 
                    D, I = faiss_index.search(np.array(embedding), k=1)
                    print("Distance:", D[0][0])
                    # Lower the distance, the better
                    if D[0][0] < 0.5:  # If a distance is above a 0.5, then the logo hasnt been seen
                        print("INDEX ALREADY EXISTS")
                        assigned_id = logo_id_map[I[0][0]]
                        logo_appearance_counts[assigned_id] += 1 # increase the amount of times weve seen this logo
                        save_frame = False
                    else:                        
                        print("ADDING NEW INDEX") # Create a new index into FAISS
                        faiss_index.add(embedding) # Add a new index (embedding)
                        logo_id_map[faiss_index.ntotal - 1] = logo_id_counter # Assign the new index to a logo_id
                        logo_appearance_counts[logo_id_counter] += 1 # increment how many times weve seen this unique ID
                        assigned_id = logo_id_counter # current assigned ID
                        logo_id_counter += 1 # Go to the next unique ID
                        save_frame = True
                    
                draw_bb_box(bbox, frame, assigned_id)

                if save_frame:
                    save_dir = "new_logo_frames"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"frame_{frame_idx}_logo_{logo_id_counter}.jpg")
                    cv2.imwrite(save_path, frame)

        processed_frames.append(frame)
        frame_idx += 1 # next frame

    cap.release()

    video_bytes = io.BytesIO()

    with imageio.get_writer(video_bytes, format='mp4', fps=fps) as writer:
        for frame in processed_frames:
            writer.append_data(frame) 

    video_bytes.seek(0)

    video_base64 = base64.b64encode(video_bytes.getvalue()).decode('utf-8')

    for faiss_idx, counter in logo_id_map.items():
        print(f'{faiss_idx} appeared approx {logo_appearance_counts[counter] * 5} times')

    return jsonify({
        "image": video_base64
    })

