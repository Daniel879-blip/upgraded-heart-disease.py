import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- Feature Selection ---------------- #
def bat_algorithm_feature_selection(X_train, y_train, n_bats=8, n_iterations=8):
    n_features = X_train.shape[1]
    rng = np.random.default_rng()
    population = rng.integers(0, 2, size=(n_bats, n_features))
    velocities = np.zeros((n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X_train[:, selected], y_train, test_size=0.3, stratify=y_train)
            model = KNeighborsClassifier()
            model.fit(X_tr, y_tr)
            fitness[i] = accuracy_score(y_te, model.predict(X_te))

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
            X_tr, X_te, y_tr, y_te = train_test_split(X_train[:, selected], y_train, test_size=0.3, stratify=y_train)
            model = KNeighborsClassifier()
            model.fit(X_tr, y_tr)
            score = accuracy_score(y_te, model.predict(X_te))
            if score > fitness[i]:
                population[i] = new_solution
                fitness[i] = score
                if score > fitness[best_idx]:
                    best_bat = new_solution.copy()
                    best_idx = i

    return np.where(best_bat == 1)[0]

def cfs_feature_selection(X_train_df, y_train, k=6):
    correlations = [abs(np.corrcoef(X_train_df.iloc[:, i], y_train)[0, 1]) for i in range(X_train_df.shape[1])]
    return np.argsort(correlations)[-k:]

# ---------------- Classifier ---------------- #
def get_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
    elif name == "Random Forest":  # ✅ Realistic settings
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=42
        )
    elif name == "SVM":
        return SVC(kernel="rbf", C=10, gamma=0.1, probability=True)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=7, weights='distance')
    return LogisticRegression()

# ---------------- Training ---------------- #
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred, zero_division=0) * 100,
        recall_score(y_test, y_pred, zero_division=0) * 100,
        f1_score(y_test, y_pred, zero_division=0) * 100,
        confusion_matrix(y_test, y_pred)
    )

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# Sidebar
st.sidebar.header("⚙️ Settings Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
classifier_name = st.sidebar.selectbox("🤖 Choose Classifier", ["Logistic Regression", "Random Forest", "SVM", "KNN"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection", ["None", "BAT", "CFS"])
scale_data = st.sidebar.checkbox("📏 Apply Feature Scaling (SVM & LR)", True)

# Load Data
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

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Dataset Preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Prepare Data
X_df = df.drop("target", axis=1)
y = df["target"].values
X = X_df.values

# Scaling
scaler = None
if scale_data and classifier_name in ["SVM", "Logistic Regression"]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Split Data First
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_df = pd.DataFrame(X_train, columns=X_df.columns)

# Feature Selection AFTER Split
if feature_method == "BAT":
    st.info("🔎 Running BAT Feature Selection (on training set only)...")
    selected_idx = bat_algorithm_feature_selection(X_train, y_train)
    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]
elif feature_method == "CFS":
    st.info("🔎 Running CFS Feature Selection (on training set only)...")
    selected_idx = cfs_feature_selection(X_train_df, y_train)
    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]

# Train Button
if st.button("🚀 Train Model"):
    with st.spinner("Training model... Please wait ⏳"):
        time.sleep(1)
        model = get_classifier(classifier_name)
        acc, prec, rec, f1, cm = train_and_evaluate(X_train, X_test, y_train, y_test, model)

    # Accuracy Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc,
        title={'text': "Accuracy (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if acc >= 80 else "orange" if acc >= 50 else "red"}}
    ))
    st.plotly_chart(gauge)

    # Metrics
    st.metric("Precision", f"{prec:.2f}%")
    st.metric("Recall", f"{rec:.2f}%")
    st.metric("F1 Score", f"{f1:.2f}%")

    # Confusion Matrix
    with st.expander("📉 Confusion Matrix"):
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

    # ✅ BAT vs CFS Comparison Chart
    if feature_method == "None":
        st.subheader("⚡ BAT vs CFS Feature Selection Comparison")

        # Run BAT
        bat_idx = bat_algorithm_feature_selection(X_train, y_train)
        X_train_bat, X_test_bat = X_train[:, bat_idx], X_test[:, bat_idx]
        bat_acc, _, _, _, _ = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test, get_classifier(classifier_name))

        # Run CFS
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train[:, cfs_idx], X_test[:, cfs_idx]
        cfs_acc, _, _, _, _ = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test, get_classifier(classifier_name))

        # Plot Comparison
        fig_sel, ax_sel = plt.subplots()
        sns.barplot(x=["BAT", "CFS"], y=[bat_acc, cfs_acc], palette="viridis", ax=ax_sel)
        ax_sel.set_ylabel("Accuracy (%)")
        st.pyplot(fig_sel)

# Real-time Prediction
st.subheader("🔍 Real-time Heart Disease Prediction")
input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
if st.button("📈 Predict Now"):
    input_df = pd.DataFrame([input_data])
    if scaler:
        input_df = scaler.transform(input_df)
    model = get_classifier(classifier_name)
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    result = "Positive (Risk of Heart Disease)" if prediction == 1 else "Negative (No Risk)"
    st.success(f"Prediction: {result}")
