import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="🌍 Earth Vision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .classification-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .class-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .class-card:hover {
        transform: translateY(-5px);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Downloading AI model... (only once)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

# Class labels with emojis and descriptions
class_info = {
    'Cloudy': {'emoji': '☁️', 'color': '#87CEEB', 'description': 'Cloud formations and weather systems'},
    'Desert': {'emoji': '🏜️', 'color': '#F4A460', 'description': 'Arid landscapes and sandy terrain'},
    'Green_Area': {'emoji': '🌿', 'color': '#90EE90', 'description': 'Vegetation, forests, and fertile land'},
    'Water': {'emoji': '🌊', 'color': '#1E90FF', 'description': 'Oceans, rivers, and water bodies'}
}

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🛰️ Earth Vision AI")
    st.markdown("Advanced satellite image classification powered by deep learning")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Information")
    st.markdown("- **Architecture**: Convolutional Neural Network")
    st.markdown("- **Input Size**: 256x256 pixels")
    st.markdown("- **Classes**: 4 environmental categories")
    st.markdown("- **Accuracy**: Optimized for satellite imagery")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 🎯 Classification Types")
    for class_name, info in class_info.items():
        st.markdown(f"**{info['emoji']} {class_name}**")
        st.markdown(f"_{info['description']}_")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-header">
    <h1>🌍 Earth Vision AI</h1>
    <p>Satellite Image Classification System</p>
    <p>Upload satellite imagery to identify environmental features using advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🔧 Loading AI model..."):
    model = download_and_load_model()

st.success("✅ AI model loaded successfully!")

# Upload section
st.markdown("""
<div class="upload-section">
    <h2>📤 Upload Your Satellite Image</h2>
    <p>Supported formats: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite image to classify environmental features"
    )

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size
    image_resized = image.resize((256, 256))
    
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(image, caption=f"Original Size: {original_size[0]}x{original_size[1]}", use_container_width=True)
        
        # Image info
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Image Information")
        st.markdown(f"**Filename**: {uploaded_file.name}")
        st.markdown(f"**Size**: {original_size[0]} × {original_size[1]} pixels")
        st.markdown(f"**File Size**: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Preprocess for prediction
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    with st.spinner("🤖 AI is analyzing your image..."):
        prediction = model.predict(img_array)[0]
        predicted_class = list(class_info.keys())[np.argmax(prediction)]
        confidence = np.max(prediction)
    
    with col2:
        # Results
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Classification Result")
        st.markdown(f"## {class_info[predicted_class]['emoji']} {predicted_class}")
        st.markdown(f"### Confidence: {confidence * 100:.1f}%")
        st.markdown(f"_{class_info[predicted_class]['description']}_")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown("### 📊 Confidence Level")
        progress_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
        st.progress(confidence, text=f"{confidence * 100:.1f}%")
        
        # Detailed predictions
        st.markdown("### 📈 Detailed Predictions")
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(class_info.keys()),
                y=prediction * 100,
                marker_color=[class_info[cls]['color'] for cls in class_info.keys()],
                text=[f"{pred*100:.1f}%" for pred in prediction],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence by Class",
            xaxis_title="Environmental Class",
            yaxis_title="Confidence (%)",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation based on confidence
        if confidence > 0.9:
            st.success("🎉 Excellent! Very high confidence in classification.")
        elif confidence > 0.7:
            st.info("👍 Good classification with solid confidence.")
        elif confidence > 0.5:
            st.warning("⚠️ Moderate confidence. Consider using a clearer image.")
        else:
            st.error("❌ Low confidence. Please try with a different image.")

# Classification examples
st.markdown("---")
st.markdown("### 🌟 Classification Examples")

examples_col1, examples_col2, examples_col3, examples_col4 = st.columns(4)

with examples_col1:
    st.markdown(f"""
    <div class="class-card">
        <h3>{class_info['Cloudy']['emoji']} Cloudy</h3>
        <p>{class_info['Cloudy']['description']}</p>
    </div>
    """, unsafe_allow_html=True)

with examples_col2:
    st.markdown(f"""
    <div class="class-card">
        <h3>{class_info['Desert']['emoji']} Desert</h3>
        <p>{class_info['Desert']['description']}</p>
    </div>
    """, unsafe_allow_html=True)

with examples_col3:
    st.markdown(f"""
    <div class="class-card">
        <h3>{class_info['Green_Area']['emoji']} Green Area</h3>
        <p>{class_info['Green_Area']['description']}</p>
    </div>
    """, unsafe_allow_html=True)

with examples_col4:
    st.markdown(f"""
    <div class="class-card">
        <h3>{class_info['Water']['emoji']} Water</h3>
        <p>{class_info['Water']['description']}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
    <h3>🚀 Earth Vision AI</h3>
    <p>Powered by TensorFlow & Streamlit | Environmental Intelligence through AI</p>
</div>
""", unsafe_allow_html=True)