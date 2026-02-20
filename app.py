from flask import Flask, render_template, Response, request, jsonify
import cv2
import math
import time
from ultralytics import YOLO

app = Flask(__name__)

print("[CV] Loading YOLO locally...")
model = YOLO("yolov8n.pt")
print("[CV] Model ready.")

# Webcam control
camera = None
camera_running = False

# Settings (controlled from dashboard)
DISTANCE_THRESHOLD = 200
ISOLATION_TIME = 20

isolation_start_time = {}

def generate_frames():
    global camera, camera_running, DISTANCE_THRESHOLD

    while camera_running:
        success, frame = camera.read()
        if not success:
            break

        height, width, _ = frame.shape
        results = model(frame, stream=True)

        centers = []
        boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if cx < 40 or cx > width - 40:
                        continue

                    centers.append((cx, cy))
                    boxes.append((x1, y1, x2, y2, cx, cy))

        isolated_count = 0
        current_time = time.time()

        for i, (x1, y1, x2, y2, cx, cy) in enumerate(boxes):

            min_distance = float("inf")
            for j, (cx2, cy2) in enumerate(centers):
                if i == j:
                    continue
                distance = math.sqrt((cx2 - cx)**2 + (cy2 - cy)**2)
                min_distance = min(min_distance, distance)

            if len(centers) == 1:
                min_distance = float("inf")

            person_id = (cx // 20, cy // 20)

            if min_distance > DISTANCE_THRESHOLD:
                if person_id not in isolation_start_time:
                    isolation_start_time[person_id] = current_time

                elapsed = current_time - isolation_start_time[person_id]

                if elapsed > ISOLATION_TIME:
                    color = (0, 0, 255)
                    isolated_count += 1
                    label = "Low Interaction"
                else:
                    color = (0, 255, 255)
                    label = f"Timer: {int(elapsed)}s"
            else:
                color = (0, 255, 0)
                label = "Engaged"
                if person_id in isolation_start_time:
                    del isolation_start_time[person_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Distance: {DISTANCE_THRESHOLD}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_running
    if not camera_running:
        camera = cv2.VideoCapture(0)
        camera_running = True
    return jsonify({"status": "camera_started"})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_running
    camera_running = False
    if camera:
        camera.release()
    return jsonify({"status": "camera_stopped"})


@app.route('/set_distance', methods=['POST'])
def set_distance():
    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = int(request.json["distance"])
    return jsonify({"status": "updated", "distance": DISTANCE_THRESHOLD})


if __name__ == "__main__":
    app.run(debug=True)