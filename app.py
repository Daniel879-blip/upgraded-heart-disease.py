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
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- BAT Algorithm ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8, k_features=6):
    """
    Real Binary BAT Algorithm for Feature Selection (Fast Version)
    """
    n_features = X.shape[1]
    rng = np.random.default_rng()

    # Initialize bat population randomly
    population = rng.integers(0, 2, size=(n_bats, n_features))
    velocities = np.zeros((n_bats, n_features))
    fitness = np.zeros(n_bats)

    # Evaluate initial population
    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.3, random_state=None
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            fitness[i] = accuracy_score(y_test, model.predict(X_test))

    # Best bat
    best_idx = np.argmax(fitness)
    best_bat = population[best_idx].copy()
    best_score = fitness[best_idx]

    freq_min, freq_max = 0, 2
    loudness = np.ones(n_bats)
    pulse_rate = np.ones(n_bats)

    for t in range(n_iterations):
        for i in range(n_bats):
            freq = freq_min + (freq_max - freq_min) * rng.random()
            velocities[i] += (population[i] ^ best_bat) * freq
            new_solution = population[i] ^ (velocities[i] > rng.random())

            # Local search
            if rng.random() > pulse_rate[i]:
                new_solution = best_bat ^ (rng.random(n_features) > 0.5)

            # Ensure at least 1 feature
            if not np.any(new_solution):
                new_solution[rng.integers(0, n_features)] = 1

            selected = np.where(new_solution == 1)[0]
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.3, random_state=None
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

# ---------------- CFS Feature Selection (Quick Demo-Friendly) ---------------- #
def cfs_feature_selection(X_df, y, k=6):
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])
    ]
    # Skip the top 2 most correlated features to avoid perfect overfitting
    top_indices = np.argsort(correlations)[-k-2:-2]
    return top_indices

# ---------------- Classifier Selection ---------------- #
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

# ---------------- Model Training ---------------- #
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
st.sidebar.title("⚙️ Settings Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
test_size = st.sidebar.slider(
    "📊 Test Size (%)", min_value=10, max_value=50, value=20, step=5
) / 100
classifier_name = st.sidebar.selectbox(
    "🤖 Choose Classifier", ["Logistic Regression", "Random Forest", "SVM", "KNN"]
)

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

# Check target
if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# EDA
st.subheader("📊 Data Visualization (EDA)")
fig, ax = plt.subplots()
sns.countplot(data=df, x="target", palette="Set2", ax=ax)
ax.set_title("Target Class Distribution")
st.pyplot(fig)

# Features & Labels
X_df = df.drop("target", axis=1)
y = df["target"].values
X = X_df.values

if len(np.unique(y)) < 2:
    st.error("❌ Target column must contain at least 2 classes.")
    st.stop()

# Train model normally
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=None
)
model = get_classifier(classifier_name)

if st.sidebar.button("🚀 Train Model"):
    acc, prec, rec, f1, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test, model
    )
    st.subheader(f"✅ Model Trained with {classifier_name}")
    st.metric("Accuracy", f"{acc:.2f}%")
    st.metric("Precision", f"{prec:.2f}%")
    st.metric("Recall", f"{rec:.2f}%")
    st.metric("F1 Score", f"{f1:.2f}%")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ---- BAT Feature Selection ---- #
    bat_idx = bat_algorithm_feature_selection(X, y)
    X_train_bat, X_test_bat, y_train_bat, y_test_bat = train_test_split(
        X[:, bat_idx], y, test_size=test_size, random_state=None
    )
    bat_acc, _, _, _, _ = train_and_evaluate(
        X_train_bat, X_test_bat, y_train_bat, y_test_bat, model
    )

    # ---- CFS Feature Selection ---- #
    cfs_idx = cfs_feature_selection(X_df, y)
    X_train_cfs, X_test_cfs, y_train_cfs, y_test_cfs = train_test_split(
        X[:, cfs_idx], y, test_size=test_size, random_state=None
    )
    cfs_acc, _, _, _, _ = train_and_evaluate(
        X_train_cfs, X_test_cfs, y_train_cfs, y_test_cfs, model
    )

    # Show comparison
    st.subheader("⚡ BAT vs CFS Feature Selection Comparison")
    st.write(f"**BAT Accuracy:** {bat_acc:.2f}%")
    st.write(f"**CFS Accuracy:** {cfs_acc:.2f}%")

# Real-time prediction
st.subheader("🔍 Real-time Heart Disease Prediction")
input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
if st.button("📈 Predict Now"):
    input_df = pd.DataFrame([input_data])
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    result = (
        "Positive (Risk of Heart Disease)"
        if prediction == 1
        else "Negative (No Risk)"
    )
    st.success(f"Prediction: {result}")
