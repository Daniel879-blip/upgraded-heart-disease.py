import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# ================= BAT Feature Selection ================= #
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

# ================= CFS Feature Selection ================= #
def cfs_feature_selection(X_df, y, k=6):
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1])
        for i in range(X_df.shape[1])
    ]
    return np.argsort(correlations)[-k:]

# ================= Train and Evaluate ================= #
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
        model
    )

# ================= Streamlit Setup ================= #
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")

# Sidebar Controls
st.sidebar.title("⚙️ Settings Panel")
st.sidebar.markdown("Adjust the model and analysis options here:")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"], help="Upload your heart disease dataset (must have a 'target' column).")
classifier_choice = st.sidebar.selectbox("🤖 Choose Classifier", ["K-Nearest Neighbour (KNN)"], help="Currently, only KNN is available for this project.")
feature_method = st.sidebar.selectbox("🧠 Feature Selection", ["Both", "BAT", "CFS"], help="Select which feature selection algorithm to use.")
k_value = st.sidebar.slider("🔢 K Value for KNN", min_value=1, max_value=15, value=7, help="Number of nearest neighbors used by the KNN algorithm.")
test_size = st.sidebar.slider("📊 Test Size (%)", min_value=10, max_value=50, value=20, step=5, help="Percentage of dataset used for testing.")

show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_feature_importance = st.sidebar.checkbox("🏅 Show Feature Importance", True)

run_analysis = st.sidebar.button("🚀 Train Model & Compare", help="Click to train the model and view comparative results.")

# Load Dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")
    st.subheader("📄 Default Dataset Loaded (heart.csv)")

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Data Prep
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# Store trained model globally
trained_model = None

# ================= Run Analysis ================= #
if run_analysis:
    with st.spinner("Training models... Please wait ⏳"):
        time.sleep(1)
        results = {}

        if feature_method in ["BAT", "Both"]:
            bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
            X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
            bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, model = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test, k_value)
            results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]
            trained_model = model

        if feature_method in ["CFS", "Both"]:
            cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
            X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
            cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, model = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test, k_value)
            results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]
            trained_model = model

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=[method], y=[results[method][0]], name=f"{method} Accuracy"))
        fig.update_layout(title="Accuracy Comparison (%)", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig)

    # Metrics Chart
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=metrics, y=results[method][1:4], name=method))
        fig.update_layout(title="Precision / Recall / F1 Score Comparison (%)")
        st.plotly_chart(fig)

    # Confusion Matrices
    if show_confusion:
        for method in results:
            st.subheader(f"{method} Confusion Matrix")
            sns.heatmap(results[method][4], annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt.gcf())

    # Feature Importance
    if show_feature_importance:
        for method in results:
            st.subheader(f"{method} Selected Features")
            st.write(list(X_df.columns[results[method][5]]))

# ================= Real-Time Prediction ================= #
st.subheader("🔍 Real-Time Heart Disease Prediction")
st.markdown("Enter patient details to predict heart disease risk.")

# Patient info inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.number_input("Number of Major Vessels Colored", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encoding
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"Yes": 1, "No": 0}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

patient_data = pd.DataFrame([[
    age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs],
    restecg_map[restecg], thalach, exang_map[exang], oldpeak,
    slope_map[slope], ca, thal_map[thal]
]], columns=X_df.columns)

# Predict button
if st.button("📈 Predict Now"):
    with st.spinner("Making prediction... 🔍"):
        time.sleep(1)

        # Auto-train model if not trained yet
        if trained_model is None:
            trained_model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
            trained_model.fit(X_train_full, y_train)

        input_scaled = scaler.transform(patient_data)
        prediction = trained_model.predict(input_scaled)[0]
        proba = trained_model.predict_proba(input_scaled)[0]

        result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {max(proba)*100:.2f}%")
