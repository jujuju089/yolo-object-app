from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            results = model(filepath)

            objects = []
            for r in results:
                for c in r.boxes.cls:
                    objects.append(model.names[int(c)])

            result = list(set(objects))
            filename = file.filename

if __name__ == "__main__":
    import random
    # zufälliger Port zwischen 3000–9000
    port = random.randint(3000, 9000)
    print(f"Server läuft auf http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
