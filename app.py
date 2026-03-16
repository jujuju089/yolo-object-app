import cv2
from ultralytics import YOLO

# YOLO Modell laden
model = YOLO("yolov8n.pt")

# Webcam starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Objekte erkennen
    results = model(frame)

    # Ergebnisse zeichnen
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
