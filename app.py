import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from handler import load_data, preprocess_data, transform_patient_input
from mode1 import bat_algorithm, cfs_algorithm, get_confusion_matrix, get_roc_curve

# ================= Page Config =================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Use the sidebar to choose feature selection and model parameters. Enter patient details below to predict risk.")

# ================= Sidebar =================
st.sidebar.header("🔧 Model Configuration")
classifier_option = st.sidebar.selectbox("Select Classifier", ["KNN"])
k_value = st.sidebar.slider("Select K Value (for KNN)", 1, 15, 7)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
feature_selector = st.sidebar.selectbox("Feature Selection Method", ["None", "BAT", "CFS"])
show_charts = st.sidebar.checkbox("Show Charts/Graphs", value=True)

# ================= Load & Preprocess Data =================
df = load_data()
X_df, scaler, X_train, X_test, y_train, y_test = preprocess_data(df, test_size=test_size)

selected_idx = None
if feature_selector == "BAT":
    selected_idx = bat_algorithm(X_train, y_train)
elif feature_selector == "CFS":
    selected_idx = cfs_algorithm(X_train, y_train)

if selected_idx is not None:
    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]

# ================= Train Model =================
model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ================== Evaluation Metrics ==================
st.subheader("📊 Model Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

# ================== Charts ==================
if show_charts:
    st.subheader("📈 Visualizations")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**ROC Curve**")
        get_roc_curve(y_test, y_proba)
    with col6:
        st.markdown("**Confusion Matrix**")
        get_confusion_matrix(y_test, y_pred)

# ================== Writeup ==================
st.markdown("---")
st.subheader("📝 About This App")
st.write("""
This app uses a K-Nearest Neighbors (KNN) classifier to predict the presence of heart disease based on patient details.
You can adjust model settings and feature selection methods from the sidebar.

**Feature Selection Methods:**
- **None**: All features are used.
- **BAT**: Metaheuristic feature selection using Binary Bat Algorithm.
- **CFS**: Correlation-Based Feature Selection.

**Prediction** is made in real-time when patient details are entered below.
""")

# ================= Real-Time Prediction ================= #
st.subheader("🔍 Real-Time Heart Disease Prediction")
st.markdown("Enter patient details to predict heart disease risk.")

with st.form("patient_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    submit_button = st.form_submit_button("📈 Predict Now")

if submit_button:
    # Mappings
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    # Create patient input DataFrame
    patient_data = pd.DataFrame([[
        age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs],
        restecg_map[restecg], thalach, exang_map[exang], oldpeak,
        slope_map[slope], ca, thal_map[thal]
    ]], columns=X_df.columns)

    input_scaled = scaler.transform(patient_data)

    # === Use trained model from BAT/CFS if available === #
    if run_analysis and "BAT" in results:
        selected_idx = results["BAT"][5]
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train_full[:, selected_idx], y_train)
        prediction = model.predict(input_scaled[:, selected_idx])[0]
        proba = model.predict_proba(input_scaled[:, selected_idx])[0]
    elif run_analysis and "CFS" in results:
        selected_idx = results["CFS"][5]
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train_full[:, selected_idx], y_train)
        prediction = model.predict(input_scaled[:, selected_idx])[0]
        proba = model.predict_proba(input_scaled[:, selected_idx])[0]
    else:
        # Fallback: train on full data
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train_full, y_train)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

    # === Output Results with Confidence === #
    if prediction == 1:
        st.error(f"🛑 Positive (Heart Disease) — Confidence: {proba[1]*100:.2f}%")
    else:
        st.success(f"✅ Negative (No Heart Disease) — Confidence: {proba[0]*100:.2f}%")

    st.write("**Confidence Scores:**")
    st.write(f"- Negative: {proba[0]*100:.2f}%")
    st.write(f"- Positive: {proba[1]*100:.2f}%")
