from PIL import Image
import cv2
import faiss
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

from utils.logo_detection_utils import *

# FOR GENERAL VIDEO SEARCH
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


# FOR SPECIFIC VIDEO SEARCH
def process_video_specific(input_video_path, reference_image_path, votes_needed=2, frame_skip=5):
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

    reference_logos, _ = extract_logo_regions(reference_image_path, save_crop=False)

    # Get the reference logo embeddings
    # Store it as a dict. Key: Model_name, value: [Vector]
    # Value is a list because the reference logo can have multiple logos
    reference_embeddings = {type(model).__name__: [] for model in embedding_models}
    for ref_logo in reference_logos:
        ref_logo = Image.fromarray(ref_logo)
        for model in embedding_models:
            reference_embeddings[type(model).__name__].append(model.extract_embedding(ref_logo))

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
                # Get the three embeddings of logo found
                input_logo = Image.fromarray(input_logo)
                input_embeddings = {type(model).__name__: model.extract_embedding(input_logo) for model in embedding_models}
                
                embedding_faiss_cmp = input_embeddings['BEiTEmbedding'].copy() # copy the beit embedding from input_embeddings
                faiss.normalize_L2(embedding_faiss_cmp) # normalize the embedding. Works really well with FAISS

                if faiss_index.ntotal == 0: # First entry into FAISS
                    
                    if verify_vote(input_embeddings, reference_embeddings, votes_needed, embedding_models):# Need to check if logo passes vote
                        print("VOTE PASSED! ADDING NEW INDEX") # Create a new index into FAISS
                        faiss_index.add(embedding_faiss_cmp) # Get embedding
                        logo_id_map[0] = logo_id_counter # First unique logo
                        logo_appearance_counts[logo_id_counter] += 1 # increment the unique logo
                        logo_id_counter += 1 # Go to next unique ID
                        assigned_id = logo_id_counter - 1 # current unique ID
                        save_frame = True

                        # draw the bounding box in the frame
                        draw_bb_box(bbox, frame, assigned_id)
                    else:
                        save_frame = False
                        print("VOTE FAILED >:(")
                        
                else: # FAISS IS NOT EMPTY
                    
                    # Get the L2 distance and index 
                    D, I = faiss_index.search(np.array(embedding_faiss_cmp), k=1)
                    print("Distance:", D[0][0])
                    # Lower the distance, the better
                    
                    if D[0][0] < 0.5:  # If a distance is above a 0.5, then the logo hasnt been seen
                        # Nothing needs to be done here for specific search
                        # If an index already exists within FAISS, then the logo has already been checked and verified by the vote
                        print("INDEX ALREADY EXISTS")
                        assigned_id = logo_id_map[I[0][0]]
                        logo_appearance_counts[assigned_id] += 1 # increase the amount of times weve seen this logo
                        save_frame = False

                        # draw the vounind box in the frame
                        draw_bb_box(bbox, frame, assigned_id)
                    else:
                        
                        if verify_vote(input_embeddings, reference_embeddings, votes_needed, embedding_models):# Need to check if logo passes vote
                            # Votes passed! Adding new index
                            print("VOTE PASSED! ADDING NEW INDEX") # Create a new index into FAISS
                            faiss_index.add(embedding_faiss_cmp) # Add a new index (embedding)
                            logo_id_map[faiss_index.ntotal - 1] = logo_id_counter # Assign the new index to a logo_id
                            logo_appearance_counts[logo_id_counter] += 1 # increment how many times weve seen this unique ID
                            assigned_id = logo_id_counter # current assigned ID
                            logo_id_counter += 1 # Go to the next unique ID
                            save_frame = True

                            # Draw the bounding box in the frame
                            draw_bb_box(bbox, frame, assigned_id)
                        else:
                            save_frame = False
                            print("VOTE FAILED >:(")
                    
                if save_frame:
                    save_dir = "new_logo_frames"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"frame_{frame_idx}_logo_{logo_id_counter}.jpg")
                    cv2.imwrite(save_path, frame)

                    input_logo_np = np.array(input_logo)
                    _, buffer = cv2.imencode('.jpg', input_logo_np)
                    logo_b64 = base64.b64encode(buffer).decode('utf-8')

                    # x1, y1, x2, y2 = bbox
                    saved_frame_data.append({
                        "frame_idx": frame_idx,
                        "logo_id": assigned_id,
                        "logo_base64": logo_b64
                    })


        out.write(frame)  # write processed frame to output

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