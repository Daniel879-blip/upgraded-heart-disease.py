# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from handler import load_data, preprocess_data
from model import (
    bat_algorithm_feature_selection,
    cfs_feature_selection,
    train_and_evaluate,
    get_roc_curve,
    predict_new
)

# ========== PAGE CONFIGURATION ========== #
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# ========== SIDEBAR CONFIG ========== #
st.sidebar.title("🔧 Model Configuration")

feature_method = st.sidebar.selectbox("Feature Selection Method", ["BAT", "CFS", "Both"])
k_value = st.sidebar.slider("K Value for KNN", 1, 15, 5)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

st.sidebar.markdown("---")
show_charts = st.sidebar.checkbox("Show Charts", value=True)
show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=True)
show_roc = st.sidebar.checkbox("Show ROC Curve", value=True)

# ========== MAIN TITLE ========== #
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the **Heart Disease Prediction App**.  
This app uses the **K-Nearest Neighbors (KNN)** algorithm to predict the likelihood of heart disease.

You can choose between two **feature selection algorithms**:
- **BAT (Bio-inspired metaheuristic)**: Optimizes feature subset based on classification performance.
- **CFS (Correlation-based Feature Selection)**: Selects top features based on correlation with target.

You can also test real-time prediction by entering patient data manually.
""")

# ========== LOAD DATA ========== #
uploaded_file = st.file_uploader("Upload Your Heart Disease Dataset (CSV)", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ========== PREPROCESS DATA ========== #
    X_df, scaler, X_train, X_test, y_train, y_test = preprocess_data(df, test_size=test_size)

    # ========== FEATURE SELECTION ========== #
    if feature_method == "BAT":
        selected_idx = bat_algorithm_feature_selection(X_train, y_train)
        st.info(f"BAT selected {len(selected_idx)} features.")
    elif feature_method == "CFS":
        selected_idx = cfs_feature_selection(X_df, df["target"])
        st.info(f"CFS selected {len(selected_idx)} features.")
    else:  # Both
        bat_idx = bat_algorithm_feature_selection(X_train, y_train)
        cfs_idx = cfs_feature_selection(X_df, df["target"])
        selected_idx = list(set(bat_idx).union(set(cfs_idx)))
        st.info(f"BAT+CFS combined selected {len(selected_idx)} features.")

    selected_columns = X_df.columns[selected_idx]
    st.write("🧬 Selected Features:", list(selected_columns))

    # ========== TRAIN AND EVALUATE ========== #
    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]

    results = train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, k_value)
    model = results["model"]

    if show_metrics:
        st.subheader("📈 Model Evaluation Metrics")
        st.write(f"**Accuracy:** {results['accuracy']}%")
        st.write(f"**Precision:** {results['precision']}%")
        st.write(f"**Recall:** {results['recall']}%")
        st.write(f"**F1 Score:** {results['f1']}%")

    if show_charts:
        st.subheader("📌 Confusion Matrix")
        cm = results["conf_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    if show_roc:
        st.subheader("🔍 ROC Curve")
        fpr, tpr, roc_auc = get_roc_curve(model, X_test_selected, y_test)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

    # ========== REAL-TIME PREDICTION ========== #
    st.subheader("💉 Real-Time Prediction")

    user_input = []
    st.markdown("Enter the patient's clinical information:")

    for col in X_df.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input.append(val)

    if st.button("Predict"):
        prediction = predict_new(model, scaler, selected_idx, user_input)
        if prediction == 1:
            st.error("⚠️ Prediction: High risk of heart disease.")
        else:
            st.success("✅ Prediction: Low risk of heart disease.")
