import os
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# YOLO Modell laden
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # YOLO Objekterkennung
            results = model(filepath)

            objects = []
            for r in results:
                for c in r.boxes.cls:
                    objects.append(model.names[int(c)])

            result = list(set(objects))
            filename = file.filename

    return render_template("index.html", result=result, filename=filename)

# ------------------------------------------------------------
# START DER APP
# ------------------------------------------------------------

if __name__ == "__main__":
    import random  # <- unbedingt 4 Leerzeichen eingerückt

    # zufälliger Port zwischen 3000–9000
    port = random.randint(3000, 9000)

    print(f"Server läuft auf http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
