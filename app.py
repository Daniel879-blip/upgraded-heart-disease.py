import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ---------------- BAT Algorithm ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8, k_features=6):
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
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.3, random_state=None, stratify=y
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            fitness[i] = accuracy_score(y_test, model.predict(X_test))

    best_idx = np.argmax(fitness)
    best_bat = population[best_idx].copy()
    best_score = fitness[best_idx]

    freq_min, freq_max = 0, 2
    loudness = np.ones(n_bats)
    pulse_rate = np.ones(n_bats)

    for _ in range(n_iterations):
        for i in range(n_bats):
            freq = freq_min + (freq_max - freq_min) * rng.random()
            velocities[i] += (population[i] ^ best_bat) * freq
            new_solution = population[i] ^ (velocities[i] > rng.random())

            if rng.random() > pulse_rate[i]:
                new_solution = best_bat ^ (rng.random(n_features) > 0.5)

            if not np.any(new_solution):
                new_solution[rng.integers(0, n_features)] = 1

            selected = np.where(new_solution == 1)[0]
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.3, random_state=None, stratify=y
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))

            if score > fitness[i] and rng.random() < loudness[i]:
                population[i] = new_solution
                fitness[i] = score
                if score > best_score:
                    best_bat = new_solution.copy()
                    best_score = score

    return np.where(best_bat == 1)[0]

# ---------------- CFS Feature Selection ---------------- #
def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    top_indices = np.argsort(correlations)[-k:]
    return top_indices

# ---------------- Classifier Selection ---------------- #
def get_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    elif name == "SVM":
        return SVC(kernel="rbf", C=10, gamma=0.1, probability=True)
    elif name == "KNN":
        return KNeighborsClassifier()
    return LogisticRegression()

# ---------------- Model Training ---------------- #
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred, zero_division=0) * 100,
        recall_score(y_test, y_pred, zero_division=0) * 100,
        f1_score(y_test, y_pred, zero_division=0) * 100,
        confusion_matrix(y_test, y_pred),
        y_pred
    )

def cross_val_accuracy(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean() * 100

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# Sidebar
st.sidebar.title("⚙️ Settings Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
classifier_name = st.sidebar.selectbox("🤖 Choose Classifier", ["Logistic Regression", "Random Forest", "SVM", "KNN"])

# Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    try:
        df = pd.read_csv("heart.csv")
        st.info("ℹ️ Using default dataset: heart.csv")
    except FileNotFoundError:
        st.error("❌ Default dataset not found. Please upload your CSV.")
        st.stop()

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# EDA
st.subheader("📊 Data Visualization (EDA)")
fig, ax = plt.subplots()
sns.countplot(data=df, x="target", palette="Set2", ax=ax)
st.pyplot(fig)

# Features & Labels
X_df = df.drop("target", axis=1)
y = df["target"].values
X = X_df.values

if len(np.unique(y)) < 2:
    st.error("❌ Target column must contain at least 2 classes.")
    st.stop()

# Scale for SVM & Logistic Regression
scaler = None
if classifier_name in ["SVM", "Logistic Regression"]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
model = get_classifier(classifier_name)
acc, prec, rec, f1, cm, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, model)
cv_acc = cross_val_accuracy(X, y, model)

st.metric("Accuracy", f"{acc:.2f}%")
st.metric("Cross-Validation Accuracy", f"{cv_acc:.2f}%")
st.metric("Precision", f"{prec:.2f}%")
st.metric("Recall", f"{rec:.2f}%")
st.metric("F1 Score", f"{f1:.2f}%")

# Confusion Matrix
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# Feature Importance
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances[sorted_idx], y=X_df.columns[sorted_idx])
    plt.title("Feature Importance")
    st.pyplot(plt)

# ROC Curve
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)

# BAT vs CFS
bat_idx = bat_algorithm_feature_selection(X, y)
bat_acc = cross_val_accuracy(X[:, bat_idx], y, model)

cfs_idx = cfs_feature_selection(X_df, y)
X_cfs = X[:, cfs_idx]
cfs_acc = cross_val_accuracy(X_cfs, y, model)

plt.figure(figsize=(5, 4))
sns.barplot(x=["BAT", "CFS"], y=[bat_acc, cfs_acc], palette="viridis")
plt.ylabel("Accuracy (%)")
plt.title("BAT vs CFS Accuracy Comparison")
st.pyplot(plt)

# Real-time Prediction
st.subheader("🔍 Real-time Heart Disease Prediction")
input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
if st.button("📈 Predict Now"):
    input_df = pd.DataFrame([input_data])
    if scaler:
        input_df = scaler.transform(input_df)
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    result = "Positive (Risk of Heart Disease)" if prediction == 1 else "Negative (No Risk)"
    st.success(f"Prediction: {result}")
