import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO

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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    resized_img = image.resize((256, 256))

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📷 Original Image", use_container_width=True)
        st.markdown(f"**🧾 Size:** `{width} x {height}` pixels")

    with col2:
        st.image(resized_img, caption="📏 Resized Image (256×256)", use_container_width=True)

    # Preprocessing
    img_array = img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Results
    st.markdown("### 🔍 Classification Result")
    st.success(f"🏷️ Predicted Class: `{predicted_label}`")
    st.progress(float(confidence))
    st.info(f"📊 Confidence Score: `{confidence * 100:.2f}%`")

    # Class probabilities chart
    st.markdown("#### 📈 Class Probability Chart")
    prob_chart = {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_chart)

    # Downloadable result
    result_text = f"""Prediction Result
----------------------
Image Size: {width} x {height}
Predicted Class: {predicted_label}
Confidence: {confidence * 100:.2f}%

Class Probabilities:
"""
    for i, cls in enumerate(CLASS_NAMES):
        result_text += f"{cls}: {prediction[i] * 100:.2f}%\n"

    result_bytes = BytesIO(result_text.encode("utf-8"))
    st.download_button(
        label="📥 Download Result as TXT",
        data=result_bytes,
        file_name="prediction_result.txt",
        mime="text/plain"
    )

else:
    st.info("📥 Please upload an image from the sidebar to begin.")
