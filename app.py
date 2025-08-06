# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

from handler import load_data, preprocess_data, train_model_with_feature_selection
from model import bat_algorithm_feature_selection, cfs_feature_selection

# ========== Streamlit Page Setup ==========
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# ========== Sidebar ==========
st.sidebar.title("⚙️ Settings")
feature_method = st.sidebar.selectbox("🧠 Feature Selection", ["BAT", "CFS", "None"])
k_value = st.sidebar.slider("K Value (KNN)", 1, 15, 5)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100
show_distribution = st.sidebar.checkbox("Show Age Distribution Plot", True)
show_pairplot = st.sidebar.checkbox("Show Pair Plot", False)
show_roc = st.sidebar.checkbox("Show ROC Curve", True)
run_training = st.sidebar.button("🚀 Train Model")

# ========== Load Data ==========
df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ========== Preprocess ==========
X_df, X_scaled, y, scaler = preprocess_data(df)

# ========== Model Training ==========
if run_training:
    model, selected_idx = train_model_with_feature_selection(
        X_df, X_scaled, y, method=feature_method, k_value=k_value, test_size=test_size
    )

    # ROC Curve
    if show_roc:
        st.subheader("📈 ROC Curve")
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled[:, selected_idx], y, test_size=test_size, stratify=y, random_state=42
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**  
        - **ROC Curve** shows the trade-off between sensitivity and specificity.  
        - **AUC (Area Under Curve)** closer to **1** means better classification performance.
        """)

    # Distribution Plot
    if show_distribution:
        st.subheader("📊 Age Distribution by Heart Disease")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df, x='age', hue='target', multiple='stack', palette='coolwarm', ax=ax)
        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**  
        - This shows the distribution of patients' ages, grouped by heart disease status.  
        - Peaks in red indicate age ranges with more heart disease cases.
        """)

    # Pair Plot
    if show_pairplot:
        st.subheader("🔗 Feature Relationships (Pair Plot)")
        st.pyplot(sns.pairplot(df[['age', 'chol', 'thalach', 'target']], hue='target').fig)
        st.markdown("""
        **Interpretation:**  
        - Visualizes feature relationships.  
        - Diagonal: feature distributions  
        - Off-diagonal: scatter plots by target
        """)

# ========== Real-Time Prediction ==========
st.subheader("🔍 Real-Time Prediction")
st.markdown("Enter patient details below:")

with st.form("patient_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    predict_btn = st.form_submit_button("📈 Predict Now")

if predict_btn:
    if 'model' not in locals() or 'selected_idx' not in locals():
        st.warning("⚠️ Please train the model first using the sidebar.")
    else:
        # Map inputs
        sex_map = {"Male": 1, "Female": 0}
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        fbs_map = {"Yes": 1, "No": 0}
        restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        exang_map = {"Yes": 1, "No": 0}
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

        patient = pd.DataFrame([[
            age, sex_map[sex], cp_map[cp], trestbps, chol,
            fbs_map[fbs], restecg_map[restecg], thalach,
            exang_map[exang], oldpeak, slope_map[slope], ca, thal_map[thal]
        ]], columns=X_df.columns)

        scaled_input = scaler.transform(patient)
        input_selected = scaled_input[:, selected_idx]
        pred = model.predict(input_selected)[0]
        prob = model.predict_proba(input_selected)[0][pred]

        if pred == 1:
            st.error(f"🛑 **Positive for Heart Disease** — Confidence: {prob * 100:.2f}%")
        else:
            st.success(f"✅ **Negative for Heart Disease** — Confidence: {prob * 100:.2f}%")
