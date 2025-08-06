import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from handler import (
    load_data,
    preprocess_data,
    train_model_with_feature_selection,
    predict_patient
)

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# === Page Title ===
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application uses **K-Nearest Neighbors (KNN)** with feature selection techniques (BAT or CFS)
to predict the likelihood of heart disease. It also offers data visualizations and real-time predictions.
""")

# === Load and Show Dataset ===
df = load_data()
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# === Preprocessing ===
X_df, X_scaled, y, scaler = preprocess_data(df)

# === Sidebar Options ===
st.sidebar.header("⚙️ Model Configuration")
k_value = st.sidebar.slider("K Value for KNN", min_value=1, max_value=15, value=7)
fs_method = st.sidebar.radio("Feature Selection Method", ("BAT", "CFS", "None"))

# === Train the Model ===
st.subheader("📊 Model Training and Evaluation")
results = train_model_with_feature_selection(X_df, X_scaled, y, method=fs_method, k_value=k_value)

metrics = results["metrics"]
model = results["model"]
selected_idx = results["selected_idx"]

st.markdown("### ✅ Evaluation Metrics")
st.write(f"**Accuracy:** {metrics['accuracy']}%")
st.write(f"**Precision:** {metrics['precision']}%")
st.write(f"**Recall:** {metrics['recall']}%")
st.write(f"**F1 Score:** {metrics['f1']}%")

# === Charts ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Confusion Matrix")
    cm = metrics["conf_matrix"]
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

with col2:
    st.markdown("#### 📉 ROC Curve")
    y_prob = model.predict_proba(X_scaled[:, selected_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

# === Extra Plots ===
st.subheader("📊 Data Visualizations")

st.markdown("#### 🔍 Age Distribution by Target")
fig_age, ax_age = plt.subplots()
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True, ax=ax_age)
st.pyplot(fig_age)

st.markdown("#### 🔍 Pair Plot of Key Features")
selected_columns = ["age", "trestbps", "chol", "thalach", "target"]
fig_pair = sns.pairplot(df[selected_columns], hue="target")
st.pyplot(fig_pair)

# === Real-Time Prediction ===
st.subheader("🧠 Real-Time Prediction")
st.markdown("Enter patient data to predict if they are likely to have heart disease.")

input_data = {}
for col in df.drop("target", axis=1).columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction, proba = predict_patient(model, scaler, selected_idx, input_df)

    st.write("### 🩺 Prediction Result")
    if prediction == 1:
        st.success("This patient is **likely to have heart disease**.")
    else:
        st.success("This patient is **unlikely to have heart disease**.")
    
    st.write(f"**Prediction Probability:** {proba[prediction]*100:.2f}%")

# === Footer ===
st.markdown("---")
st.markdown("© 2025 - Built for Heart Disease Risk Analysis using Machine Learning.")
