import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- Feature Selection ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    n_features = X.shape[1]
    rng = np.random.default_rng()
    population = rng.integers(0, 2, size=(n_bats, n_features))
    velocities = np.zeros((n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(X[:, selected], y, test_size=0.3, stratify=y)
            model = KNeighborsClassifier(n_neighbors=7, weights='distance')
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
            X_train, X_test, y_train, y_test = train_test_split(X[:, selected], y, test_size=0.3, stratify=y)
            model = KNeighborsClassifier(n_neighbors=7, weights='distance')
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            if score > fitness[i]:
                population[i] = new_solution
                fitness[i] = score
                if score > fitness[best_idx]:
                    best_bat = new_solution.copy()
                    best_idx = i

    return np.where(best_bat == 1)[0]

def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    return np.argsort(correlations)[-k:]

# ---------------- Training Function ---------------- #
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred, zero_division=0) * 100,
        recall_score(y_test, y_pred, zero_division=0) * 100,
        f1_score(y_test, y_pred, zero_division=0) * 100,
        confusion_matrix(y_test, y_pred)
    )

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="Comparative Analysis - BAT vs CFS on KNN", layout="wide")
st.title("🩺 Comparative Analysis of BAT and CFS Feature Selection on KNN Classifier")

st.markdown("### Project Focus")
st.info("This app compares **BAT (Bat Algorithm)** and **CFS (Correlation-Based Feature Selection)** using **KNN Classifier** on the Heart Disease dataset.")

# File upload
uploaded_file = st.file_uploader("📁 Upload Heart Disease Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    try:
        df = pd.read_csv("heart.csv")
        st.info("ℹ️ Using default dataset: heart.csv")
    except FileNotFoundError:
        st.error("❌ No dataset found. Please upload one.")
        st.stop()

# Check target column
if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Dataset preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Target distribution
st.subheader("📊 Target Class Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="target", palette="Set2", ax=ax)
st.pyplot(fig)

# Feature correlation heatmap
st.subheader("📈 Feature Correlation Heatmap")
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
st.pyplot(plt)

# Prepare data
X_df = df.drop("target", axis=1)
y = df["target"].values
X = X_df.values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split once for consistency
X_train_full, X_test_full, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Training button
if st.button("🚀 Run Comparative Analysis"):
    with st.spinner("Running BAT Feature Selection..."):
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test)

    with st.spinner("Running CFS Feature Selection..."):
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test)

    st.success("✅ Analysis Completed!")

    # Comparison Table
    st.subheader("📊 Performance Comparison Table")
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        "BAT": [bat_acc, bat_prec, bat_rec, bat_f1],
        "CFS": [cfs_acc, cfs_prec, cfs_rec, cfs_f1]
    })
    st.dataframe(comparison_df)

    # Comparison Bar Chart
    st.subheader("📉 Performance Comparison Chart")
    fig, ax = plt.subplots()
    width = 0.35
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    bat_scores = [bat_acc, bat_prec, bat_rec, bat_f1]
    cfs_scores = [cfs_acc, cfs_prec, cfs_rec, cfs_f1]
    x = np.arange(len(metrics))
    ax.bar(x - width/2, bat_scores, width, label="BAT")
    ax.bar(x + width/2, cfs_scores, width, label="CFS")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    st.pyplot(fig)

    # Gauge for Accuracy
    st.subheader("🎯 Accuracy Gauges")
    col1, col2 = st.columns(2)
    with col1:
        gauge_bat = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bat_acc,
            title={'text': "BAT Accuracy (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
        ))
        st.plotly_chart(gauge_bat)
    with col2:
        gauge_cfs = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cfs_acc,
            title={'text': "CFS Accuracy (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
        ))
        st.plotly_chart(gauge_cfs)

    # Confusion Matrices
    st.subheader("📌 Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**BAT Confusion Matrix**")
        fig_bat, ax_bat = plt.subplots()
        sns.heatmap(bat_cm, annot=True, fmt="d", cmap="Blues", ax=ax_bat)
        st.pyplot(fig_bat)
    with col2:
        st.markdown("**CFS Confusion Matrix**")
        fig_cfs, ax_cfs = plt.subplots()
        sns.heatmap(cfs_cm, annot=True, fmt="d", cmap="Greens", ax=ax_cfs)
        st.pyplot(fig_cfs)

# Real-Time Prediction
st.subheader("🔍 Real-time Heart Disease Prediction (KNN)")
input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
if st.button("📈 Predict Now"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train_full, y_train)
    prediction = model.predict(input_scaled)[0]
    result = "Positive (Risk of Heart Disease)" if prediction == 1 else "Negative (No Risk)"
    st.success(f"Prediction: {result}")
