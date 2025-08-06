import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ============ BAT Feature Selection ============
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    n_features = X.shape[1]
    rng = np.random.default_rng(42)
    population = rng.integers(0, 2, size=(n_bats, n_features))
    velocities = np.zeros((n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.2, stratify=y, random_state=42
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            fitness[i] = accuracy_score(y_test, model.predict(X_test))

    best_idx = np.argmax(fitness)
    best_bat = population[best_idx].copy()

    for _ in range(n_iterations):
        for i in range(n_bats):
            freq = rng.random()
            velocities[i] += (population[i] ^ best_bat) * freq
            new_solution = population[i] ^ (velocities[i] > rng.random())
            if not np.any(new_solution):
                new_solution[rng.integers(0, n_features)] = 1
            selected = np.where(new_solution == 1)[0]
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.2, stratify=y, random_state=42
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            if score > fitness[i]:
                population[i] = new_solution
                fitness[i] = score
                if score > fitness[best_idx]:
                    best_bat = new_solution.copy()
                    best_idx = i

    return np.where(best_bat == 1)[0]

# ============ CFS Feature Selection ============
def cfs_feature_selection(X_df, y, k=6):
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1])
        for i in range(X_df.shape[1])
    ]
    return np.argsort(correlations)[-k:]

# ============ Train & Evaluate ============
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        confusion_matrix(y_test, y_pred),
        model,
        y_pred
    )

# ============ Streamlit Setup ============
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")

# Sidebar Controls
st.sidebar.title("⚙️ Settings Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection", ["Both", "BAT", "CFS"])
k_value = st.sidebar.slider("🔢 K Value for KNN", min_value=1, max_value=15, value=7)
test_size = st.sidebar.slider("📊 Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_feature_importance = st.sidebar.checkbox("🏅 Show Feature Importance", True)
show_roc = st.sidebar.checkbox("📊 Show ROC Curve", True)
run_analysis = st.sidebar.button("🚀 Train Model & Compare")

# Load Dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")
    st.subheader("📄 Default Dataset Loaded (heart.csv)")

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Show dataset preview
st.dataframe(df.head())

# Data Prep
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# Store trained model
trained_model = None

# ============ Run Analysis ============
if run_analysis:
    results = {}
    
    if feature_method in ["BAT", "Both"]:
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model, _ = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test, k_value)
        results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]
        trained_model = bat_model  # Store for real-time prediction
    
    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model, _ = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test, k_value)
        results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]
        trained_model = cfs_model  # Store for real-time prediction

    # Charts and explanations here...

# ============ Real-Time Prediction ============
st.subheader("🔍 Real-Time Heart Disease Prediction")
st.markdown("Enter patient details to predict heart disease risk.")

# Patient info form (same as your current one)
# ... (KEEP EXACTLY SAME AS YOU HAVE IT NOW)

# Fix prediction issue: use trained_model
if st.button("📈 Predict Now"):
    if trained_model is None:
        st.error("⚠️ Please train the model first before making predictions.")
    else:
        input_df = pd.DataFrame([[...]], columns=X_df.columns)  # Keep your existing patient form conversion
        input_scaled = scaler.transform(input_df)
        prediction = trained_model.predict(input_scaled)[0]
        proba = trained_model.predict_proba(input_scaled)[0]
        result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {max(proba)*100:.2f}%")
