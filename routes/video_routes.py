from flask import Blueprint, request
from flask_cors import CORS

video_blueprint = Blueprint("video", __name__, url_prefix="/video")
CORS(video_blueprint)

@video_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    return


@video_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    return