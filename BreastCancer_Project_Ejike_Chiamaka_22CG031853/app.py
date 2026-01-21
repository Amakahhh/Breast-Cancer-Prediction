"""
Breast Cancer Prediction System - Streamlit Application
Author: Ejike Chiamaka (22CG031853)
Algorithm: Support Vector Machine (SVM)
Model Persistence: Joblib
EDUCATIONAL PURPOSE ONLY - NOT FOR MEDICAL DIAGNOSIS
"""

import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Breast Cancer Prediction System")
st.write("""
**Author:** Ejike Chiamaka (22CG031853)  
**Algorithm:** Support Vector Machine (SVM)  
**Model:** Joblib-persisted

⚠️ **EDUCATIONAL PURPOSE ONLY** - Not for medical diagnosis
""")

st.divider()

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'model')
        
        # Load files
        model = joblib.load(os.path.join(model_dir, 'breast_cancer_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        features = joblib.load(os.path.join(model_dir, 'selected_features.pkl'))
        
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model, scaler, selected_features = load_artifacts()
st.success(f"✅ Model loaded | Features: {', '.join(selected_features)}")

st.divider()

# Input Section
st.subheader("📊 Enter Tumor Measurements")

col1, col2 = st.columns(2)

with col1:
    f1 = st.number_input("Mean Radius", 0.0, value=15.0, step=0.1)
    f2 = st.number_input("Mean Texture", 0.0, value=20.0, step=0.1)
    f3 = st.number_input("Mean Area", 0.0, value=500.0, step=1.0)

with col2:
    f4 = st.number_input("Mean Smoothness", 0.0, 1.0, 0.1, 0.001)
    f5 = st.number_input("Mean Compactness", 0.0, 1.0, 0.15, 0.001)

st.divider()

# Predict
if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[f1, f2, f3, f4, f5]])
    input_scaled = scaler.transform(input_data)
    
    pred_label = int(model.predict(input_scaled)[0])
    proba = model.predict_proba(input_scaled)[0]
    confidence = float(np.max(proba)) * 100
    
    st.divider()
    st.subheader("🎯 Prediction Result")
    
    if pred_label == 1:
        st.success("### ✅ BENIGN (Non-cancerous)")
        st.metric("Confidence", f"{confidence:.2f}%")
    else:
        st.error("### ⚠️ MALIGNANT (Cancerous)")
        st.metric("Confidence", f"{confidence:.2f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Benign %", f"{proba[1]*100:.2f}%")
    with col2:
        st.metric("Malignant %", f"{proba[0]*100:.2f}%")
    
    st.divider()
    
    summary = {
        'Feature': selected_features,
        'Input': [f1, f2, f3, f4, f5],
        'Scaled': [f"{x:.4f}" for x in input_scaled[0]]
    }
    st.table(summary)

st.divider()
st.markdown("⚕️ **DISCLAIMER:** Educational purposes only. Not for medical use. Consult healthcare professionals.")


