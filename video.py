import cv2
import numpy as np
import streamlit as st
import tempfile
import requests
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO  # Import YOLOv8 model

# Load ResNet50 model
model = load_model('models/Resnet50Model.h5')
class_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 
    'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 
    'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]  # Update with actual class names

# Load YOLOv8 model for sign detection
yolo_model = YOLO('yolov8n.pt')  # Use a YOLOv8 pre-trained model

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to detect traffic signs using YOLOv8
def detect_signs_with_yolo(frame):
    if frame is None:
        print("Error: Failed to capture frame.")
        return None
    
    results = yolo_model(frame)  # Detect objects in the frame
    return results

# Streamlit App
st.title("🚦 Smart Traffic Sign Detection and Classifier System")
option = st.radio("Select Input", ["Upload Video", "Phone Camera Stream", "Laptop Webcam"])

if option == "Upload Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Frame capture failed.")
                break  # Exit the loop if frame capture fails

            sign_detections = detect_signs_with_yolo(frame)
            if sign_detections:
                for detection in sign_detections[0].boxes:
                    x1, y1, x2, y2 = map(int, detection.xywh[0])  # Get bounding box coordinates
                    cropped_sign = frame[y1:y2, x1:x2]
                    if cropped_sign.size > 0:  # Check if the cropped sign is not empty
                        preprocessed_image = preprocess_image(cropped_sign)
                        predictions = model.predict(preprocessed_image)
                        detected_class = class_names[np.argmax(predictions)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle for non-traffic signs
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif option == "Phone Camera Stream":
    stream_url = st.text_input("Enter your phone's IP camera stream URL")
    if st.button("Start Camera Stream") and stream_url:
        cap = cv2.VideoCapture(f'http://admin:admin@{stream_url.split("//")[-1]}')
        frame_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame from stream.")
                break  # Exit if frame capture fails
            sign_detections = detect_signs_with_yolo(frame)
            if sign_detections:
                for detection in sign_detections[0].boxes:
                    x1, y1, x2, y2 = map(int, detection.xywh[0])  # Get bounding box coordinates
                    cropped_sign = frame[y1:y2, x1:x2]
                    if cropped_sign.size > 0:
                        preprocessed_image = preprocess_image(cropped_sign)
                        predictions = model.predict(preprocessed_image)
                        detected_class = class_names[np.argmax(predictions)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle for non-traffic signs
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif option == "Laptop Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame from webcam.")
                break  # Exit if frame capture fails
            sign_detections = detect_signs_with_yolo(frame)
            if sign_detections:
                for detection in sign_detections[0].boxes:  # Adjust based on YOLO result format
                    x1, y1, x2, y2 = map(int, detection.xywh[0])  # Get bounding box coordinates
                    cropped_sign = frame[y1:y2, x1:x2]
                    if cropped_sign.size > 0:
                        preprocessed_image = preprocess_image(cropped_sign)
                        predictions = model.predict(preprocessed_image)
                        detected_class = class_names[np.argmax(predictions)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle for non-traffic signs
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
