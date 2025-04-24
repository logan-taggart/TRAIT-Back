import cv2
import faiss
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import os
from collections import defaultdict
import numpy as np
import subprocess
import imageio_ffmpeg

from utils.embed import BEiTEmbedding, CLIPEmbedding, ResNetEmbedding
beit = BEiTEmbedding()
clip = CLIPEmbedding()
resnet = ResNetEmbedding()

embedding_models = [beit, clip, resnet]

from flask import jsonify
import base64


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

    results = model(img,conf=0.6)
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


def add_logo_to_faiss(faiss_index, embedding, logo_id_map, logo_id_counter):
    '''
    Add a new logo embedding to the FAISS index and update the ID map
    Assumes the the embedduing is already normalized
    '''
    # Add the embedding to the FAISS index (already normalized)
    faiss_index.add(embedding)
    # Update the logo ID map with the new logo ID
    logo_id_map[faiss_index.ntotal - 1] = logo_id_counter
    # Increment the logo ID counter for the next unique logo
    return logo_id_counter + 1

def update_logo_in_faiss(faiss_index, embedding, logo_id_map, logo_appearance_counts, threshold=0.5):
    '''Search FAISS index for a logo and update counts if a match is found or a new entry is added'''

    save_frame = False
    # Get the distance and index of the nearest neighbor
    D, I = faiss_index.search(np.array(embedding), k=1)
    # Check if the distance is less than the threshold (LOWER IS BETTER)
    if D[0][0] < threshold:
        # Logo already exists
        assigned_id = logo_id_map[I[0][0]]
        # Increment the count of appearances for this logo
        logo_appearance_counts[assigned_id] += 1
        save_frame = False
    else:
        # New logo seen, add it to FAISS
        assigned_id = add_logo_to_faiss(faiss_index, embedding, logo_id_map, len(logo_id_map))
        # Set the number of times we seen this new logo to 1 (first appearance)
        logo_appearance_counts[assigned_id] = 1
        save_frame = True
    return assigned_id, save_frame

def process_video(input_video_path, frame_skip=5):
    # resnet size = 2048
    # clip size = 512
    # beit size = 768
    embedding_dim = 768 # Size for BEIT
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    logo_id_counter = 0 # How many unique logos we've seen
    logo_id_map = {}  # maps FAISS index to logo ID
    logo_appearance_counts = defaultdict(int) # How many times a unique logo has appeared
    
    # Remove any existing processed video files
    if os.path.exists("./processed_videos/processed_video.mp4"):
        os.remove("./processed_videos/processed_video.mp4")


    output_video_dir = "./processed_videos"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = "./processed_videos/temp_processed_video.mp4"
    temp_compressed_path = "./processed_videos/processed_video.mp4"

    cap = cv2.VideoCapture(input_video_path)
    # This will give an error, but it still works. GO WITH IT
    # Actually doesnt support avc1, so the video size is like 900mb.
    # The reason we need avc1 is because we need the H264 cocec for the video to play within the application
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    save_frame = False
    saved_frame_data = []

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
                # Get the logo region in rgb format
                embedding = embedding_models[0].extract_embedding(input_logo)
                faiss.normalize_L2(embedding) # normalize the embedding. Works really well with FAISS

                # Process the embedding with FAISS
                # Checks if the logo already exists in the FAISS index
                assigned_id, save_frame = update_logo_in_faiss(faiss_index, embedding, logo_id_map, logo_appearance_counts)
                    
                draw_bb_box(bbox, frame, assigned_id)

                if save_frame:
                    save_dir = "new_logo_frames"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"frame_{frame_idx}_logo_{logo_id_counter}.jpg")
                    cv2.imwrite(save_path, frame)

                    _, buffer = cv2.imencode('.jpg', input_logo)
                    logo_b64 = base64.b64encode(buffer).decode('utf-8')

                    # x1, y1, x2, y2 = bbox
                    saved_frame_data.append({
                        "frame_idx": frame_idx,
                        "logo_id": assigned_id,
                        "logo_base64": logo_b64
                    })

        out.write(frame)
        frame_idx += 1 # next frame

    cap.release()
    out.release()

    # Lower CRF = Higher Quality and Less Compression
    # Higher CRF = Lower Quality and More Compression
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", output_video_path,
        "-vcodec", "libx264", "-crf", "23", "-preset", "ultrafast",
        temp_compressed_path
    ])

    os.replace(output_video_path, temp_compressed_path)

    return jsonify({
        # "video": video_path,
        "saved_frames": saved_frame_data,
        "logo_appearance_count": logo_appearance_counts
    })

