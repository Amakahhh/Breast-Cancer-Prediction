# Breast Cancer Prediction System

**Author:** Ejike Chiamaka  
**Matric Number:** 22CG031853  
**Algorithm:** Support Vector Machine (SVM)  
**Model Persistence:** Joblib  
**Deployment Platform:** Streamlit Cloud

## Project Overview

This is an educational machine learning system that predicts whether a breast tumor is **benign** or **malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset. The system employs a Support Vector Machine (SVM) classifier with RBF kernel.

⚠️ **DISCLAIMER:** This system is strictly for educational purposes and must NOT be presented as a medical diagnostic tool. Always consult qualified medical professionals for medical advice.

---

## Selected Features (5 Input Features)

1. **Mean Radius** - Average distance from center to perimeter points
2. **Mean Texture** - Standard deviation of gray-scale values  
3. **Mean Area** - Tumor size measurement
4. **Mean Smoothness** - Local radius variation (consistency)
5. **Mean Compactness** - Perimeter² / area - 1.0 (shape indicator)

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9035 (90.35%) |
| **Precision** | 0.9552 |
| **Recall** | 0.8889 |
| **F1-Score** | 0.9209 |

---

## Project Structure

```
BreastCancer_Project_Ejike_Chiamaka_22CG031853/
├── app.py                          # Streamlit web application (CORRECTED inference logic)
├── requirements.txt                # Python dependencies
├── BreastCancer_hosted_webGUI_link.txt  # Submission file with deployment link
├── /model/
│   ├── model_building.ipynb       # Jupyter notebook with model development
│   ├── breast_cancer_model.pkl    # Trained SVM model (Joblib)
│   ├── scaler.pkl                 # StandardScaler (Joblib)
│   └── selected_features.pkl      # Feature list (Joblib)
├── /static/ (optional)
│   └── style.css
└── /.streamlit/
    └── config.toml                # Streamlit configuration
```

---

## Key Fixes Applied (Avoiding Previous Mistakes)

### ✅ Data Preprocessing (No Leakage)
- **BEFORE:** ❌ Scaler was fit BEFORE train-test split
- **AFTER:** ✅ Train-test split FIRST, then fit scaler on training data only
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)     # Fit on training
X_test_scaled = scaler.transform(X_test)          # Apply to test
```

### ✅ Model Inference Logic (Correct predict/proba)
- **BEFORE:** ❌ Used `np.argmax()` and `np.max()` on SVM.predict() output
- **AFTER:** ✅ Use `predict()` for label and `predict_proba()` for confidence
```python
pred_label = int(model.predict(features_scaled)[0])
probabilities = model.predict_proba(features_scaled)[0]
confidence = float(np.max(probabilities)) * 100
```

### ✅ Deployment Configuration (Environment Variables)
- **BEFORE:** ❌ Hard-coded `app.run(debug=True, port=5003)`
- **AFTER:** ✅ Uses environment variables via Streamlit config
```toml
[server]
port = ${PORT:-8501}
headless = true
```

### ✅ Feature Order Consistency
- Saved `selected_features.pkl` to ensure model receives features in correct order
- App.py validates input order matches training order

### ✅ Health Check Endpoint
- Added system status verification to catch deployment issues early
- Returns model load status and system readiness

---

## Installation & Setup

### Local Development

1. **Clone/Download the project**
   ```bash
   cd BreastCancer_Project_Ejike_Chiamaka_22CG031853
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Access the web interface**
   - Open browser to: `http://localhost:8501`

### Deployment on Streamlit Cloud

1. **Push project to GitHub**
   - Create GitHub repo with project structure
   - Push all files including `model/` directory

2. **Connect to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select repository, branch, and main file (`app.py`)
   - Click "Deploy"

3. **Access deployed app**
   - Streamlit generates live URL automatically
   - Share URL with users

---

## Model Building Process

The `model_building.ipynb` notebook includes:

1. **Data Loading** - Breast Cancer Wisconsin dataset (569 samples, 30 features)
2. **Missing Value Check** - Verified no missing values
3. **Feature Selection** - Selected 5 features from approved list
4. **Train-Test Split** - 80% train, 20% test (BEFORE scaling)
5. **Feature Scaling** - StandardScaler fitted on training data only
6. **Model Training** - SVM with RBF kernel and probability=True
7. **Evaluation** - Accuracy, Precision, Recall, F1-Score
8. **Model Persistence** - Saved using Joblib (model + scaler + features)
9. **Reloading Test** - Verified model can be reloaded and used for prediction

---

## Usage Instructions

### Using the Web Interface

1. **Enter Tumor Characteristics**
   - Input all 5 feature values (mean radius, texture, area, smoothness, compactness)
   - Values should be within typical ranges from dataset

2. **Click "Predict"**
   - System scales input using trained scaler
   - Runs through SVM model
   - Returns prediction (Benign/Malignant) with confidence percentage

3. **Interpret Results**
   - **Green (Benign):** Low risk, non-cancerous tumor predicted
   - **Red (Malignant):** High risk, cancerous tumor predicted
   - Confidence score indicates model certainty

### API Usage (Streamlit)

The app provides:
- **Health Check:** Verifies model is loaded
- **Input Validation:** Ensures features are in correct order
- **Probability Output:** Shows confidence for both classes

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found error** | Verify `model/` folder exists with `*.pkl` files |
| **Port already in use** | Change port in `.streamlit/config.toml` |
| **Import errors** | Run `pip install -r requirements.txt` |
| **Scaling mismatch** | Verify scaler was fitted on training data only |
| **Inconsistent predictions** | Check feature order matches `selected_features.pkl` |

---

## References

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- **scikit-learn SVM:** https://scikit-learn.org/stable/modules/svm.html
- **Streamlit Docs:** https://docs.streamlit.io/

---

## Important Notes

✅ **Correct:** Train-test split first, then scale  
✅ **Correct:** Use `predict()` for labels, `predict_proba()` for probabilities  
✅ **Correct:** Fit scaler on training data only  
✅ **Correct:** Use environment variables for configuration  
✅ **Correct:** Save and reuse selected features list  

❌ **Avoid:** Scaling before train-test split (data leakage)  
❌ **Avoid:** Using `argmax()` on classifier output  
❌ **Avoid:** Hard-coded port/debug settings  
❌ **Avoid:** Feature order inconsistency  

---

**Educational Purpose Only** - This system is designed for learning machine learning concepts. It is not approved for clinical use.

Last Updated: January 21, 2026
