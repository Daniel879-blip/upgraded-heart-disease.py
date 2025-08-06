import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from handler import (
    load_data,
    preprocess_data,
    train_model_with_feature_selection
)
from model import predict_new

# ========== Streamlit Config ========== #
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.sidebar.title("⚙️ Settings & Controls")

# ========== Sidebar Controls ========== #
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection Method", ["Both", "BAT", "CFS"])
classifier_choice = st.sidebar.selectbox("🤖 Classifier", ["KNN"])
k_value = st.sidebar.slider("🔢 K Value for KNN", 1, 15, 7)
test_size = st.sidebar.slider("📊 Test Size (%)", 10, 50, 20, step=5) / 100
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_roc_curve = st.sidebar.checkbox("📊 Show ROC Curve", True)
show_distribution_plots = st.sidebar.checkbox("📊 Show Feature Distributions", True)
show_pairplot = st.sidebar.checkbox("🔗 Show Pair Plot", True)
run_analysis = st.sidebar.button("🚀 Train Model & Compare")

# ========== Load and Preprocess Data ========== #
df = load_data(uploaded_file)
if df is None or "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

X_df, y, scaler, X_train_full, X_test_full, y_train, y_test = preprocess_data(df, test_size)

# ========== Train & Evaluate Models ========== #
results = {}
if run_analysis:
    results = train_model_with_feature_selection(
        X_df, y, X_train_full, X_test_full, y_train, y_test,
        method=feature_method, k_value=k_value
    )

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=results[method]["accuracy"],
                title={"text": f"{method} Accuracy"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
            ))
        st.plotly_chart(fig)
        st.markdown("""
        **Interpretation:** Accuracy reflects the overall correctness of the classifier.  
        A higher percentage means better performance.
        """)

    # Precision / Recall / F1 Chart
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(
                x=metrics,
                y=[
                    results[method]["precision"],
                    results[method]["recall"],
                    results[method]["f1"]
                ],
                name=method
            ))
        fig.update_layout(title="Precision / Recall / F1 Score Comparison (%)")
        st.plotly_chart(fig)
        st.markdown("""
        **Interpretation:**  
        - **Precision**: How many predicted positives are actually positive?  
        - **Recall**: How many actual positives were identified correctly?  
        - **F1 Score**: Harmonic average of precision and recall.
        """)

    # Confusion Matrices
    if show_confusion:
        for method in results:
            st.subheader(f"{method} Confusion Matrix")
            sns.heatmap(results[method]["conf_matrix"], annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt.gcf())
            st.markdown("""
            - **TN** (Top-left): True Negatives  
            - **FP** (Top-right): False Positives  
            - **FN** (Bottom-left): False Negatives  
            - **TP** (Bottom-right): True Positives
            """)

    # ROC Curve
    if show_roc_curve:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        knn_model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        knn_model.fit(X_train_full, y_train)
        y_proba = knn_model.predict_proba(X_test_full)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='blue')
        ax.plot([0, 1], [0, 1], linestyle='--', color='red')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        st.markdown("""
        **Interpretation:** ROC Curve shows trade-off between sensitivity and specificity.  
        Higher AUC = better classifier performance.
        """)

    # Distribution Plot
    if show_distribution_plots:
        st.subheader("Feature Distribution: Age vs Heart Disease")
        fig, ax = plt.subplots()
        sns.histplot(df, x='age', hue='target', multiple='stack', palette='coolwarm', ax=ax)
        st.pyplot(fig)

    # Pair Plot
    if show_pairplot:
        st.subheader("Feature Pair Plot")
        st.pyplot(sns.pairplot(df[['age', 'chol', 'thalach', 'target']], hue='target').fig)

# ========== Real-Time Prediction ========== #
st.subheader("🔍 Real-Time Heart Disease Prediction")
st.markdown("Enter patient details to predict heart disease risk.")

with st.form("prediction_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    submit_button = st.form_submit_button("📈 Predict Now")

if submit_button:
    input_data = pd.DataFrame([[
        age,
        1 if sex == "Male" else 0,
        {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp],
        trestbps,
        chol,
        1 if fbs == "Yes" else 0,
        {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg],
        thalach,
        1 if exang == "Yes" else 0,
        oldpeak,
        {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope],
        ca,
        {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]
    ]], columns=X_df.columns)

    prediction, confidence = predict_new(
        results, feature_method, input_data, scaler, k_value,
        X_train_full, y_train
    )

    if prediction == 1:
        st.error(f"🛑 Positive for Heart Disease — Confidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Negative for Heart Disease — Confidence: {confidence:.2f}%")
