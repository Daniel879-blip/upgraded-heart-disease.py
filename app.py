import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Page settings
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# Sidebar Settings
st.sidebar.title("⚙️ Settings Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
test_size = st.sidebar.slider("📊 Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
classifier_name = st.sidebar.selectbox("🤖 Choose Classifier", ["Logistic Regression", "Random Forest", "SVM", "KNN"])

# Load Dataset
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("heart.csv")
        st.info("ℹ️ Using default dataset: heart.csv")
    except FileNotFoundError:
        st.error("❌ Default dataset not found. Please upload your CSV.")
        st.stop()

# Check for target column
if 'target' not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Data Display
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# EDA Section
st.subheader("📊 Data Visualization (EDA)")
fig, ax = plt.subplots()
sns.countplot(data=df, x='target', palette='Set2', ax=ax)
ax.set_title("Target Class Distribution")
st.pyplot(fig)

# Feature and Label
X = df.drop('target', axis=1)
y = df['target']

# Check if y has at least 2 classes
if len(np.unique(y)) < 2:
    st.error("❌ Target column must contain at least 2 classes.")
    st.stop()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Select Classifier
def get_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif name == "Random Forest":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVC()
    elif name == "KNN":
        return KNeighborsClassifier()
    return LogisticRegression()

# Train Button
if st.sidebar.button("🚀 Train Model"):
    model = get_classifier(classifier_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.subheader(f"✅ Model Trained with {classifier_name}")
    st.metric("Accuracy", f"{acc:.2f}")
    st.metric("Precision", f"{prec:.2f}")
    st.metric("Recall", f"{rec:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# Real-Time Prediction Section
st.subheader("🔍 Real-time Heart Disease Prediction")

input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"{column}", format="%.2f")

if st.button("📈 Predict Now"):
    input_df = pd.DataFrame([input_data])
    model = get_classifier(classifier_name)
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    result = "Positive (Risk of Heart Disease)" if prediction == 1 else "Negative (No Risk)"
    st.success(f"Prediction: {result}")
