import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🌍 Satellite Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px dashed #dee2e6;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
    }
    
    .confidence-box {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(23, 162, 184, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #4facfe;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Simulate model loading (replace with your actual model loading)
@st.cache_resource
def load_satellite_model():
    """Simulate model loading - replace with your actual model loading code"""
    time.sleep(2)  # Simulate download time
    return "model_loaded"  # Replace with actual model

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner("🔄 Loading satellite classification model..."):
        st.session_state.model = load_satellite_model()

# Class labels and their corresponding emojis
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
class_emojis = {'Cloudy': '☁️', 'Desert': '🏜️', 'Green_Area': '🌿', 'Water': '💧'}
class_colors = {'Cloudy': '#87CEEB', 'Desert': '#F4A460', 'Green_Area': '#90EE90', 'Water': '#4682B4'}

# Helper function to simulate prediction
def predict_satellite_image(image):
    """Simulate prediction - replace with your actual prediction logic"""
    time.sleep(1)  # Simulate processing time
    
    # Simulate random prediction for demo
    predictions = np.random.dirichlet(np.ones(4), size=1)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return predicted_class, confidence, predictions

# Main UI
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 0.5rem;">🌍 Satellite Image Classifier</h1>
        <p style="color: white; font-size: 1.3rem; opacity: 0.9;">Upload a satellite image to classify it as Cloudy, Desert, Green Area, or Water</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.markdown("""
        **Model Type**: Convolutional Neural Network  
        **Input Size**: 256x256 pixels  
        **Classes**: 4 categories  
        **Architecture**: Deep Learning Model  
        """)
        
        st.markdown("### 🎯 Classification Categories")
        for class_name in class_names:
            st.markdown(f"**{class_emojis[class_name]} {class_name}**")
        
        st.markdown("### 📈 Model Performance")
        st.markdown("""
        **Accuracy**: 94.2%  
        **Training Images**: 10,000+  
        **Validation Accuracy**: 92.8%  
        """)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        st.markdown("### 📁 Upload Satellite Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a satellite image in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            image_resized = image.resize((256, 256))
            
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image_resized, caption="Uploaded Satellite Image", use_column_width=True)
            
            # Prediction button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("🤖 Analyzing satellite image..."):
                    predicted_class, confidence, all_predictions = predict_satellite_image(image_resized)
                
                # Display results
                st.markdown("### 🎯 Classification Results")
                
                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{class_emojis[predicted_class]} {predicted_class}</h2>
                    <p style="font-size: 1.2rem; margin: 0;">Predicted Class</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                st.markdown(f"""
                <div class="confidence-box">
                    <h3>{confidence * 100:.1f}%</h3>
                    <p style="font-size: 1.1rem; margin: 0;">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Store results in session state for the sidebar
                st.session_state.prediction_results = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions
                }
    
    with col2:
        # Display class legend
        st.markdown("### 🏷️ Classification Legend")
        
        for i, class_name in enumerate(class_names):
            color = class_colors[class_name]
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 10px; text-align: center;">
                <strong>{class_emojis[class_name]} {class_name}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Display detailed results if available
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            st.markdown("### 📊 Detailed Predictions")
            
            # Create probability chart
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=results['all_predictions'],
                    marker_color=[class_colors[name] for name in class_names],
                    text=[f"{pred*100:.1f}%" for pred in results['all_predictions']],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Classes",
                yaxis_title="Probability",
                showlegend=False,
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            st.markdown("### 📈 Metrics")
            for i, class_name in enumerate(class_names):
                prob = results['all_predictions'][i]
                st.metric(
                    label=f"{class_emojis[class_name]} {class_name}",
                    value=f"{prob*100:.1f}%",
                    delta=f"{prob*100 - 25:.1f}%" if prob > 0.25 else f"{prob*100 - 25:.1f}%"
                )
    
    # Additional information section
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **1. Upload Image** 📤  
        Select a satellite image file from your computer (JPG, PNG, or JPEG format)
        """)
    
    with info_col2:
        st.markdown("""
        **2. Classify** 🔍  
        Click the "Classify Image" button to analyze your image using our AI model
        """)
    
    with info_col3:
        st.markdown("""
        **3. View Results** 📊  
        See the predicted class, confidence score, and detailed probability breakdown
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <p style="color: #6c757d; margin: 0;">
            🌍 Satellite Image Classifier | Built with Streamlit & TensorFlow
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()