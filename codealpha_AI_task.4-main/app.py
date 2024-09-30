# app.py

from flask import Flask, render_template, Response
import cv2
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model (or Faster R-CNN)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace 'yolov5s' with your preferred model

# Function to generate frames for video streaming
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Object detection
        results = model(frame)
        
        # Draw bounding boxes on the frame
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define the routes for the web app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
