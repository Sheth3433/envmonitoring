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

# -------------------- UI SETUP --------------------
st.set_page_config(
    page_title="🌍 Land Cover Classification",
    page_icon="🛰️",
    layout="wide"
)

st.sidebar.title("🛰️ Satellite Image Analyzer")
st.sidebar.markdown("Upload a satellite image and classify land cover types.")

uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

st.title("🌿 Environmental Monitoring Dashboard")

col1, col2 = st.columns([1, 2])

if uploaded_file:
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        resized_img = image.resize((256, 256))
        st.image(image, caption="📷 Original Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Classification Result")

        # Preprocess image
        img_array = img_to_array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0]
        predicted_label = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"**🏷️ Predicted Class:** `{predicted_label}`")
        st.info(f"**📊 Confidence Score:** `{confidence * 100:.2f}%`")

        # Optional: Show full class probabilities
        st.markdown("#### 📈 Class Probabilities")
        for i, prob in enumerate(prediction):
            st.write(f"🔸 {CLASS_NAMES[i]}: `{prob * 100:.2f}%`")

else:
    st.info("📥 Please upload an image from the sidebar to begin.")

