import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from gtts import gTTS
import os
import base64

st.set_page_config(page_title="🚦 Traffic Sign Recognition with Voice", layout="wide")


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
    'CNN Model': 'models/cnn_model.h5'
}

# Model selection
selected_model = st.selectbox(
    "🤖 Select Model",
    list(model_options.keys())
)

model_path = model_options[selected_model]
model = load_model(model_path)

class_labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 
    'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 
    'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

uploaded_file = st.file_uploader("📤 Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image
    image_resized = cv2.resize(image, (32, 32))
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    # Display the uploaded image
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.image(image, width=320, caption="Uploaded Traffic Sign", use_column_width=False)

    
    if st.button("🔍 Predict and Speak", type="primary"):
        with st.spinner(f"Processing with {selected_model}..."):
            prediction = model.predict(image_resized)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class]
            confidence = f"{np.max(prediction) * 100:.2f}%"
            
            explanation_text = f"The {predicted_label} sign is detected. Always follow the rules for safety."
            voice_text = f"Prediction: {predicted_label}. Explanation: {explanation_text}. Confidence Score: {confidence}."

            # Generate voice output
            tts = gTTS(voice_text)
            tts.save("prediction_voice.mp3")

            # Convert audio file to base64
            with open("prediction_voice.mp3", "rb") as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode()

            with col2:
                st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
                st.markdown(f"""
                    <h3 style='color: #22c55e;'>✅ Model Used: {selected_model}</h3>
                    <h2 style='color: #60a5fa;'>🚗 Prediction: <span style='color:#facc15;'>{predicted_label}</span></h2>
                    <p style='font-size:16px; color: #d1d5db;'>📖 <b>Explanation:</b> {explanation_text}</p>
                    <p style='font-size:15px; color:#9ca3af;'>🟢 Confidence Score: <b>{confidence}</b></p>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                """, unsafe_allow_html=True)
    else:
        st.info("📥 Please upload a traffic sign image to continue.")

# Footer
st.markdown("<div class='footer'>© 2025 Traffic Vision AI</div>", unsafe_allow_html=True)
