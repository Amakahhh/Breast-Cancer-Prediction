"""
Breast Cancer Prediction System - Streamlit Web Application
Author: Ejike Chiamaka
Matric Number: 22CG031853
Algorithm: Support Vector Machine (SVM)

This application loads a pre-trained SVM model and allows users to input tumor
feature values to predict whether a tumor is benign or malignant.

EDUCATIONAL DISCLAIMER: This system is strictly for educational purposes and
must NOT be used as a medical diagnostic tool.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stContainer {
        max-width: 600px;
        margin: 0 auto;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Page title and description
st.title("🏥 Breast Cancer Prediction System")
st.markdown("---")
st.write("""
**Developed by**: Ejike Chiamaka (22CG031853)  
**Algorithm**: Support Vector Machine (SVM)  
**Model Persistence**: Joblib  

This educational tool predicts whether a tumor is **benign** or **malignant** 
based on five key tumor features from the Breast Cancer Wisconsin dataset.

⚠️ **Educational Disclaimer**: This system is for educational purposes only 
and must NOT be used for actual medical diagnosis. Please consult a healthcare professional.
""")

st.markdown("---")

# Load model, scaler, and features
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model, scaler, and selected features."""
    try:
        # Define model directory - handle both local and deployment scenarios
        if os.path.exists('./model'):
            model_dir = './model'
        elif os.path.exists('model'):
            model_dir = 'model'
        else:
            st.error("Model directory not found!")
            st.stop()
        
        # Load model
        model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
        model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        # Load selected features
        features_path = os.path.join(model_dir, 'selected_features.pkl')
        selected_features = joblib.load(features_path)
        
        return model, scaler, selected_features
    
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        st.stop()

# Load artifacts
model, scaler, selected_features = load_model_artifacts()

# Display feature information
st.subheader("📋 Selected Features")
st.write(f"The model uses the following **5 features** for prediction:")
feature_df = pd.DataFrame({
    'Feature Name': selected_features,
    'Feature Index': range(1, len(selected_features) + 1)
})
st.dataframe(feature_df, use_container_width=True)

st.markdown("---")

# Input section
st.subheader("📝 Enter Tumor Feature Values")
st.write("Please enter the measured values for each feature:")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input(
        "1. Radius Mean",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
        help="Average distance from center to points on the perimeter"
    )
    
    texture = st.number_input(
        "2. Texture Mean",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.1,
        help="Standard deviation of gray-scale values"
    )
    
    area = st.number_input(
        "3. Area Mean",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=1.0,
        help="Average tumor area"
    )

with col2:
    smoothness = st.number_input(
        "4. Smoothness Mean",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Local variation in radius lengths"
    )
    
    compactness = st.number_input(
        "5. Compactness Mean",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="Perimeter² / area - 1.0"
    )

st.markdown("---")

# Prediction button
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # Prepare input features in correct order
    input_features = np.array([radius, texture, area, smoothness, compactness]).reshape(1, -1)
    
    # Scale features using loaded scaler (fitted on training data)
    input_scaled = scaler.transform(input_features)
    
    # Get prediction and probability using CORRECT approach
    # predict() returns class label (0 or 1)
    # predict_proba() returns probability distribution
    predicted_label = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = float(np.max(probabilities)) * 100
    
    # Display prediction result
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    if predicted_label == 1:
        prediction_text = "BENIGN"
        prediction_icon = "✅"
        css_class = "benign"
        interpretation = "The tumor is predicted to be **benign** (non-cancerous)."
    else:
        prediction_text = "MALIGNANT"
        prediction_icon = "⚠️"
        css_class = "malignant"
        interpretation = "The tumor is predicted to be **malignant** (cancerous). Please consult a healthcare professional immediately."
    
    # Create prediction display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(
            label="Prediction",
            value=prediction_text,
            delta=None,
            delta_color="off"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%",
            delta=None,
            delta_color="off"
        )
    
    st.write(interpretation)
    
    # Show detailed probabilities
    st.markdown("#### Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Classification': ['Malignant', 'Benign'],
        'Probability': [f"{probabilities[0]*100:.2f}%", f"{probabilities[1]*100:.2f}%"]
    })
    st.dataframe(prob_df, use_container_width=True)
    
    # Show input summary
    st.markdown("#### Input Values Summary")
    input_summary = pd.DataFrame({
        'Feature': selected_features,
        'Input Value': [radius, texture, area, smoothness, compactness],
        'Scaled Value': input_scaled[0]
    })
    st.dataframe(input_summary, use_container_width=True)

st.markdown("---")

# Information section
with st.expander("ℹ️ How This Model Works"):
    st.write("""
    **Algorithm**: Support Vector Machine (SVM) with RBF kernel
    
    **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
    - 569 tumor samples with 30 computed features
    - Binary classification: Benign vs Malignant
    
    **Model Training Process**:
    1. Selected 5 key features from the dataset
    2. Checked for missing values (none found)
    3. Split data into 80% training, 20% testing
    4. Applied StandardScaler normalization (fitted on training data only)
    5. Trained SVM classifier using scaled training data
    6. Evaluated with Accuracy, Precision, Recall, and F1-Score
    
    **Important Notes**:
    - The scaler was fitted only on training data to prevent data leakage
    - Feature scaling is applied before prediction
    - Model uses probability calibration for confidence scores
    - This is an educational system, not a medical diagnostic tool
    """)

with st.expander("📚 Feature Information"):
    st.write("""
    **Selected Features Description**:
    
    1. **Radius Mean**: Average distance from center to perimeter points
    2. **Texture Mean**: Standard deviation of gray-scale values
    3. **Area Mean**: Tumor size measurement
    4. **Smoothness Mean**: Local radius variation (consistency)
    5. **Compactness Mean**: Perimeter² / area - 1.0 (shape indicator)
    
    These features help distinguish between benign and malignant tumors
    based on their geometric and density characteristics.
    """)

st.markdown("---")

# Footer with important disclaimer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 15px; 
            background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
    <strong>⚕️ IMPORTANT MEDICAL DISCLAIMER</strong><br>
    This application is designed for educational purposes only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical concerns.
</div>
""", unsafe_allow_html=True)

st.markdown("""
---
**Project Information**  
Author: Ejike Chiamaka | Matric: 22CG031853 | CSC 415 - AI Course Assignment
""", unsafe_allow_html=True)
