import tempfile
import os
import io
from flask import Blueprint, request, send_file
from flask_cors import CORS

from utils.process_video import *

video_blueprint = Blueprint("video", __name__, url_prefix="/video")
CORS(video_blueprint)

@video_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    if "main_video" not in request.files:
        return {"error": "No file provided"}, 400
    
    main_video = request.files["main_video"]
    bb_color = request.form.get("bb_color")

    video_stream = io.BytesIO(main_video.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_stream.read())
        temp_video_path = tmp_file.name
    
    response = process_video(temp_video_path)

    os.remove(temp_video_path)

    return response

@video_blueprint.route("/fetch-processed-video", methods=["GET"])
def fetch_processed_video():
    video_path = './processed_videos/processed_video.mp4'

    if not os.path.exists(video_path):
        return {"error": "Video not found"}, 404

    return send_file(video_path, mimetype='video/mp4')


@video_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    # if "main_video" not in request.files or "reference_image" not in request.files:
    #     return {"error": "No file provided"}, 400
    
    # main_video = request.files["main_video"]
    # reference_image = request.files["reference_image"]
    # similarity_threshold = int(request.form.get("confidence"))
    # bb_color = request.form.get("bb_color")

    # video_stream = io.BytesIO(main_video.read())

    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
    #     tmp_file.write(video_stream.read())
    #     temp_video_path = tmp_file.name
    
    # response = process_video_specific(temp_video_path, reference_image, 2)

    # os.remove(temp_video_path)

    # return response
    return