import streamlit as st
import pandas as pd
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

# Set Streamlit page config
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict the presence of heart disease using machine learning and feature selection techniques.")

# Sidebar
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your heart.csv file", type=["csv"])

if uploaded_file is None:
    st.sidebar.warning("Please upload a CSV file to continue.")
    st.stop()

feature_method = st.sidebar.selectbox("Feature Selection Method", ["None", "BAT", "CFS"])
k_value = st.sidebar.slider("K-Value for KNN", 1, 15, 7)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

show_charts = st.sidebar.checkbox("Show Charts and Graphs", value=True)

# Load and preprocess
df = load_data(uploaded_file)
X_df, scaler, X_train, X_test, y_train, y_test = preprocess_data(df, test_size=test_size)

# Feature selection
selected_indices = list(range(X_df.shape[1]))  # default: all features
if feature_method == "BAT":
    st.sidebar.info("Using BAT for feature selection...")
    selected_indices = bat_algorithm_feature_selection(X_train, y_train)
elif feature_method == "CFS":
    st.sidebar.info("Using CFS for feature selection...")
    selected_indices = cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train)

# Apply selected features
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

# Train and evaluate
results = train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, k_value=k_value)

# Show metrics
st.subheader("📊 Model Evaluation Metrics")
st.write(f"**Accuracy:** {results['accuracy']}%")
st.write(f"**Precision:** {results['precision']}%")
st.write(f"**Recall:** {results['recall']}%")
st.write(f"**F1 Score:** {results['f1']}%")

# Confusion matrix
if show_charts:
    st.subheader("🧾 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(results["conf_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, roc_auc = get_roc_curve(results["model"], X_test_selected, y_test)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# Real-time prediction
st.subheader("🩺 Real-Time Heart Disease Prediction")

with st.form("prediction_form"):
    inputs = []
    for i in selected_indices:
        col_name = X_df.columns[i]
        value = st.number_input(f"{col_name}", value=0.0)
        inputs.append(value)
    submitted = st.form_submit_button("Predict")

    if submitted:
        pred = predict_new(results["model"], scaler, selected_indices, inputs)
        if pred[0] == 1:
            st.error("🚨 High Risk of Heart Disease Detected!")
        else:
            st.success("✅ No Heart Disease Detected.")

# Explanation section
st.markdown("---")
st.header("🧠 About the App")
st.markdown("""
This app uses a K-Nearest Neighbors (KNN) classifier to predict heart disease. 
You can optionally apply feature selection techniques:
- **BAT (Bat Algorithm):** A bio-inspired optimization method.
- **CFS (Correlation-based Feature Selection):** Selects features most correlated with the target.
         
You can also tune:
- **K-value:** Controls the number of neighbors used in KNN.
- **Test size:** Controls the train/test split ratio.

This app demonstrates real-time predictions and shows model performance metrics and visualizations.
""")
