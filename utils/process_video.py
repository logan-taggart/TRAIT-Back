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

from utils.video_progress import video_progress

from utils.logo_detection_utils import *

def setup_faiss(embedding_dim=768):
    # resnet size = 2048
    # clip size = 512
    # beit size = 768

    # Initialize FAISS index for L2 distance
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    logo_id_counter = 0 # How many unique logos we've seen
    logo_id_map = {}  # maps FAISS index to logo ID
    logo_appearance_counts = defaultdict(int) # How many times a unique logo has appeared
    return faiss_index, logo_id_counter, logo_id_map, logo_appearance_counts

def setup_directories():
    # Remove any existing processed video files
    if os.path.exists("./processed_videos/processed_video.mp4"):
        os.remove("./processed_videos/processed_video.mp4")


    # Create the output directory if it doesn't exist
    output_video_dir = "./processed_videos"
    os.makedirs(output_video_dir, exist_ok=True)
    # Where the processed video will be saved
    output_video_path = "./processed_videos/temp_processed_video.mp4"
    temp_compressed_path = "./processed_videos/processed_video.mp4"

    return output_video_path, temp_compressed_path

def setup_opencv_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    # This will give an error, but it still works. GO WITH IT
    # Actually doesnt support avc1, so the video size is like 900mb.
    # The reason we need avc1 is because we need the H264 cocec for the video to play within the application
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    return cap, out

def run_ffmpeg_subprocess(input_video_path, output_video_path):
    # Lower CRF = Higher Quality and Less Compression
    # Higher CRF = Lower Quality and More Compression
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", input_video_path,
        "-vcodec", "libx264", "-crf", "23", "-preset", "ultrafast",
        output_video_path
    ])

    # Replace the original video with the processed one
    os.replace(input_video_path, output_video_path)


# FOR GENERAL VIDEO SEARCH
# Change frame_skip possobly to speed up?
def process_video(input_video_path, bounding_box_threshold, bb_color, frame_skip=5):
    
    # Initialize FAISS, the counter for what logo ID we are on, and the logo appearance counts
    # FAISS index is used to store the logo embeddings and their corresponding IDs
    # logo_id_counter is used to assign unique IDs to logos
    # logo_id_map is used to map FAISS index to logo ID
    # logo_appearance_counts is used to count how many times each logo has appeared
    faiss_index, logo_id_counter, logo_id_map, logo_appearance_counts = setup_faiss(embedding_dim=768)
    
    output_video_path, temp_compressed_path = setup_directories()
    
    # Setup the video capture and writer to process the video
    cap, out = setup_opencv_video(input_video_path, output_video_path)
    
    # Convert the color from hex to BGR for OpenCV
    bb_color = hex_to_bgr(bb_color)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_progress['total_frames'] = total_frames

    frame_idx = 0
    saved_frame_data = []
    dynamic_frame_skip = frame_skip  # start with 5
    logo_present = False

    while cap.isOpened():
        ret, frame = cap.read()

        video_progress['progress_percentage'] = float((frame_idx / total_frames)) * 100
        if not ret:
            break  # stop if video ends

        if frame_idx % dynamic_frame_skip == 0:
            print(f"Processing frame {frame_idx}")
        
            input_logos, input_bboxes = extract_logo_regions(frame, bounding_box_threshold, save_crop=False)

            # Check if any logos are found in this frame
            if input_logos:
                if not logo_present:
                    print("Logo found. Switching to processing every frame.")
                logo_present = True
                dynamic_frame_skip = 1  # now process every frame

                for input_logo, bbox in zip(input_logos, input_bboxes):
                    embedding = embedding_models[0].extract_embedding(input_logo)
                    faiss.normalize_L2(embedding)

                    assigned_id, save_frame = update_logo_in_faiss(
                        faiss_index, embedding, logo_id_map, logo_appearance_counts
                    )

                    draw_bb_box(bbox, frame, assigned_id, bb_color=bb_color)

                    if save_frame:
                        saved_frame_data.append(
                            save_frame_func(frame, frame_idx, logo_id_counter, input_logo)
                        )
            else:
                if logo_present:
                    print(f"No logos found. Reverting to every other {frame_skip} frames.")
                logo_present = False
                dynamic_frame_skip = frame_skip  # back to 5

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    run_ffmpeg_subprocess(output_video_path, temp_compressed_path)

    return jsonify({
        "saved_frames": saved_frame_data,
        "logo_appearance_count": logo_appearance_counts
    })


# FOR SPECIFIC VIDEO SEARCH
def process_video_specific(input_video_path, reference_image_path,bounding_box_threshold, bb_color, votes_needed=2, frame_skip=5):
    
    # Initialize FAISS, the counter for what logo ID we are on, and the logo appearance counts
    # FAISS index is used to store the logo embeddings and their corresponding IDs
    # logo_id_counter is used to assign unique IDs to logos
    # logo_id_map is used to map FAISS index to logo ID
    # logo_appearance_counts is used to count how many times each logo has appeared
    faiss_index, logo_id_counter, logo_id_map, logo_appearance_counts = setup_faiss(embedding_dim=768)
    
    output_video_path, temp_compressed_path = setup_directories()

    # Setup the video capture and writer to process the video
    cap, out = setup_opencv_video(input_video_path, output_video_path)

    frame_idx = 0
    save_frame = False
    saved_frame_data = []
    dynamic_frame_skip = frame_skip  # start with 5
    logo_present = False

    reference_logos, _ = extract_logo_regions(reference_image_path, bounding_box_threshold, save_crop=False)

    # Convert the color from hex to BGR for OpenCV
    bb_color = hex_to_bgr(bb_color) 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_progress['total_frames'] = total_frames
    
    # Get the reference logo embeddings
    # Store it as a dict. Key: Model_name, value: [Vector]
    # Value is a list because the reference logo can have multiple logos
    reference_embeddings = {type(model).__name__: [] for model in embedding_models}
    for ref_logo in reference_logos:
        # convert the logo to PIL image. Do this for the ResNet model
        PIL_ref_logo = Image.fromarray(ref_logo)
        for model in embedding_models:
            # Get the embedding for each model
            # Store it as a dict. Key: Model_name, value: [Vector]
            # Value is a list because the reference logo can have multiple logos
            reference_embeddings[type(model).__name__].append(model.extract_embedding(PIL_ref_logo))

    while cap.isOpened():
        ret, frame = cap.read()

        video_progress['progress_percentage'] = float((frame_idx / total_frames)) * 100

        if not ret:
            break  # stop if video ends

        if frame_idx % dynamic_frame_skip == 0:  # process every 5th frame
            print(f"Processing frame {frame_idx}")

            # extract detected logos from the current frame
            input_logos, input_bboxes = extract_logo_regions(frame,bounding_box_threshold, save_crop=False)
            logo_matched_this_frame = False

            # draw a bounding box around each detected logo
            for input_logo, bbox in zip(input_logos, input_bboxes):
                # Get the three embeddings of logo found
                # convert the logo to PIL image. Do this for the ResNet model
                PIL_input_logo = Image.fromarray(input_logo)
                # input_logo = Image.fromarray(input_logo)
                input_embeddings = {type(model).__name__: model.extract_embedding(PIL_input_logo) for model in embedding_models}
                
                embedding_faiss_cmp = input_embeddings['BEiTEmbedding'].copy() # copy the beit embedding from input_embeddings
                faiss.normalize_L2(embedding_faiss_cmp) # normalize the embedding. Works really well with FAISS

                # Get the L2 distance and index 
                D, I = faiss_index.search(np.array(embedding_faiss_cmp), k=1)
                print("Distance:", D[0][0])
                # Lower the distance, the better
                
                if D[0][0] < 0.5:  # If a distance is above a 0.5, then the logo hasnt been seen
                    # Nothing needs to be done here for specific search
                    # If an index already exists within FAISS, then the logo has already been checked and verified by the vote
                    print("INDEX ALREADY EXISTS")
                    # increase the amount of times weve seen this logo
                    assigned_id, save_frame = increase_logo_appearance_count(logo_appearance_counts, logo_id_map, I, assigned_id) 

                    # draw the bounding box in the frame
                    draw_bb_box(bbox, frame, assigned_id, bb_color=bb_color)
                    logo_matched_this_frame = True
                else:
                    
                    if verify_vote(input_embeddings, reference_embeddings, votes_needed, embedding_models):# Need to check if logo passes vote
                        # Votes passed! Adding new index
                        print("VOTE PASSED! ADDING NEW INDEX")
                        
                        # Set the number of times we seen this new logo to 1 (first appearance)
                        logo_appearance_counts[logo_id_counter] = 1 
                        # current assigned ID is the logo_id_counter
                        assigned_id = logo_id_counter 
                        # Add the new logo to FAISS and increment the ID counter
                        logo_id_counter = add_logo_to_faiss(faiss_index, embedding_faiss_cmp, logo_id_map, logo_id_counter)
                        # We want to save this frame because it is the first time we've seen this logo
                        save_frame = True

                        # Draw the bounding box in the frame
                        draw_bb_box(bbox, frame, assigned_id, bb_color=bb_color)
                        logo_matched_this_frame = True
                    else:
                        save_frame = False
                        print("VOTE FAILED >:(")
                    
                if save_frame:
                    saved_frame_data.append(save_frame_func(frame, frame_idx, logo_id_counter, input_logo))

            if logo_matched_this_frame:
                if not logo_present:
                    print("Matching logo found. Switching to every frame.")
                logo_present = True
                dynamic_frame_skip = 1
            else:
                if logo_present:
                    print("No matching logos found. Switching back to every 5th frame.")
                logo_present = False
                dynamic_frame_skip = frame_skip
                
        out.write(frame)  # write processed frame to output

        frame_idx += 1 # next frame

    cap.release()
    out.release()

    run_ffmpeg_subprocess(output_video_path, temp_compressed_path)

    return jsonify({
        "saved_frames": saved_frame_data,
        "logo_appearance_count": logo_appearance_counts
    })