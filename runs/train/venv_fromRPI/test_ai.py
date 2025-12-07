# minimal_real_time_yolo_stream.py
import cv2
from flask import Flask, Response
from ultralytics import YOLO

# minimal_real_time_yolo_stream.py
import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('/home/pi/ConveyorBelt-mqtt/Venv/models/runs_yolo/pcb-detect/weights/best.pt')  # Use the working model path

def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 for default camera; adjust if needed
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame, conf=0.25, iou=0.45)
        annotated_frame = results[0].plot()  # Annotate frame with detections
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
