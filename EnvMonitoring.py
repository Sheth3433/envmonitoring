import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- CONFIG --------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Satellite Land Classifier", layout="centered")
st.title("🌍 Environmental Monitoring via Satellite")
st.write("Upload a satellite image to classify it as **Cloudy**, **Desert**, **Green Area**, or **Water**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"✅ Prediction: **{predicted_label}**")
    st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")
