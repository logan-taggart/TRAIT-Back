from flask import Blueprint, request
from flask_cors import CORS

from utils.detect import *

image_blueprint = Blueprint("image", __name__, url_prefix="/image")
CORS(image_blueprint)

@image_blueprint.route("/detect-all", methods=["POST"])
def detect_all():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    return process_image(file)


@image_blueprint.route("/detect-specific", methods=["POST"])
def detect_specific():
    return