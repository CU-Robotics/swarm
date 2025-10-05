#!/usr/bin/env python3
import cv2
import pipeline
import logging
import pathlib
import numpy as np
from util import TextBox
from flask import Flask, send_from_directory, request, jsonify, Response

app = Flask(__name__)

# load metadata
ctx = pipeline.get_stage_context()
rows = list(ctx.rows(filter_by={False})) # query all invalid rows
image_index = 0

frontend_path = pathlib.Path.cwd() / "frontend"

@app.route("/")
def get_index():
    return send_from_directory(frontend_path, "index.html")

@app.route("/frontend/<path:filename>")
def get_frontend(filename):
    return send_from_directory(frontend_path, filename)

@app.route("/image/current")
def get_current_image():
    data, src, _ = rows[image_index]
    image = cv2.imread(str(src))
    gamma = 2
    gain = 1.5 # multiply final intensity
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 * gain for i in range(256)])
    table = np.clip(table, 0, 255).astype("uint8")
    eq = cv2.LUT(image, table)
    with TextBox(eq, scale=3, thickness=2) as tb:
        tb.write(data["name"])
        tb.write(f"{image_index+1}/{len(rows)}", color=(0, 255, 255))
    success, buf = cv2.imencode(".png", eq, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        return "encode failed", 500
    return Response(buf.tobytes(), mimetype="image/png")

@app.route("/image/skip", methods=["POST"])
def post_next_image():
    global image_index
    image_index = (image_index + int(request.args.get("jump"))) % len(rows)
    return jsonify({"index": image_index, "total": len(rows)})

@app.route("/image/labels", methods=["GET"])
def get_image_labels():
    data, _, _ = rows[image_index]
    plates = data["labels"].get("plates", [])
    return jsonify({"plates": plates})

@app.route("/image/labels", methods=["POST"])
def post_image_labels():
    new_plates = request.get_json()
    data, _, _ = rows[image_index]
    data["labels"]["plates"] = new_plates.get("plates", [])
    return "success", 200


if __name__ == "__main__":
    # disable internal flask logger
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    
    host = "127.0.0.1"
    port = 8000

    logging.info(f"labeler running at: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)