import cv2
import numpy as np
import streamlit as st
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array



st.set_page_config(page_title="🚦 Smart Traffic Sign Detection and Classifier System", layout="wide")


st.markdown("""
    <style>
        body { background-color: transparent; }
        .header { text-align: center; color: #22c55e; font-size: 38px; font-weight: bold; margin-bottom: 12px; }
        .sub-header { text-align: center; color: #e5e7eb; font-size: 18px; margin-bottom: 40px; }
        .card { background-color: #1f2937; padding: 25px; border-radius: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); color: #f9fafb; }
        .predict-box { background-color: #111827; padding: 25px; border-radius: 12px; transition: transform 0.3s ease; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
        .predict-box:hover { transform: scale(1.02); }
        .footer { text-align: center; color: #9ca3af; font-size: 14px; margin-top: 50px;}
        select, input { color: black; }
        audio { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>🚦 Traffic Sign Classifier with Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload a traffic sign image, get prediction with voice-assisted output.</div>", unsafe_allow_html=True)

model_options = {
    'ResNet50': 'models/ResNet50model.h5',
    'CNN Model': 'models/cnn_model.h5',
    "Simple Enhanced Resnet" :  'models/final_resnet_model.h5'
}

# Model selection
selected_model = st.selectbox(
    "🤖 Select Model",
    list(model_options.keys())
)

model_path = model_options[selected_model]
model = load_model(model_path)
# Load the ResNet model

class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Path to the frozen inference graph
PATH_TO_CKPT = 'models/frozen_inference_graph.pb'

# Load the TensorFlow detection graph
detection_graph = tf.Graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95  # Adjust the fraction as needed
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

def preprocess_image(image):
    if selected_model == 'ResNet50':
        image = cv2.resize(image, (32, 32))
    else:
        image = cv2.resize(image, (160, 160))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def detect_and_classify_signs(frame):
    """Detect traffic signs with TensorFlow model and classify them with ResNet."""
    # Prepare image for TensorFlow model
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Get input and output tensors from the graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Run detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )

    # Process detections
    height, width = frame.shape[:2]
    for i in range(int(num[0])):
        score = scores[0, i]
        if score > 0.5:  # Confidence threshold
            # Convert normalized coordinates to pixel values
            ymin, xmin, ymax, xmax = boxes[0, i]
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)

            if x1 < x2 and y1 < y2:
                # Crop the detected region
                cropped_sign = frame[y1:y2, x1:x2]

                if cropped_sign.size > 0:
                    # Preprocess for ResNet
                    preprocessed_image = preprocess_image(cropped_sign)
                    
                    # Classify with ResNet
                    predictions = model.predict(preprocessed_image)
                    detected_class = class_names[np.argmax(predictions)]
                    confidence = float(np.max(predictions))

                    # Draw bounding box and classification
                    if confidence > 0.7:  # Adjustable ResNet confidence threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{detected_class} ({confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


option = st.radio("Select Input", ["Upload Video", "Phone Camera Stream", "Laptop Webcam"])

if option == "Upload Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame = detect_and_classify_signs(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif option == "Phone Camera Stream":
    stream_url = st.text_input("Enter your phone's IP camera stream URL")
    if st.button("Start Camera Stream") and stream_url:
        cap = cv2.VideoCapture(f'http://admin:admin@{stream_url.split("//")[-1]}')
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Stream")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or stop_button:
                break
            
            frame = detect_and_classify_signs(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif option == "Laptop Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Webcam")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or stop_button:
                break

            frame = detect_and_classify_signs(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# Clean up TensorFlow session when done (optional, if running in a persistent environment)
sess.close()