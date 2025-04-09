from flask import Blueprint, request
from flask_cors import CORS

video_blueprint = Blueprint("video", __name__, url_prefix="/video")
CORS(video_blueprint)

@video_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    if "main_video" not in request.files:
        return {"error": "No file provided"}, 400
    main_video = request.files["main_video"]
    bb_color = request.form.get("bb_color")
    return 


@video_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    if "main_video" not in request.files or "reference_image" not in request.files:
        return {"error": "No file provided"}, 400
    main_video = request.files["main_video"]
    reference_image = request.files["reference_image"]
    similarity_threshold = int(request.form.get("confidence"))
    bb_color = request.form.get("bb_color")
    return