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

    return render_template("index.html", result=result, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
