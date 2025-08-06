import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ================= Feature Selection (BAT & CFS) ================= #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    n_features = X.shape[1]
    rng = np.random.default_rng(42)
    population = rng.integers(0, 2, size=(n_bats, n_features))
    fitness = np.zeros(n_bats)
    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.2, stratify=y, random_state=42
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            fitness[i] = accuracy_score(y_test, model.predict(X_test))
    best_bat = population[np.argmax(fitness)].copy()
    return np.where(best_bat == 1)[0]

def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    return np.argsort(correlations)[-k:]

# ================= Train & Evaluate ================= #
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred) * 100,
        precision_score(y_test, y_pred, zero_division=0) * 100,
        recall_score(y_test, y_pred, zero_division=0) * 100,
        f1_score(y_test, y_pred, zero_division=0) * 100,
        confusion_matrix(y_test, y_pred),
        model
    )

# ================= Streamlit Setup ================= #
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings & Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection Method", ["Both", "BAT", "CFS"])
classifier_choice = st.sidebar.selectbox("🤖 Classifier", ["KNN"])
k_value = st.sidebar.slider("🔢 K Value for KNN", min_value=1, max_value=15, value=7)
test_size = st.sidebar.slider("📊 Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_feature_importance = st.sidebar.checkbox("🏅 Show Feature Importance", True)
show_roc = st.sidebar.checkbox("📈 Show ROC Curve", True)
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
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Data Prep
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# ================= Run Analysis ================= #
if run_analysis:
    results = {}
    if feature_method in ["BAT", "Both"]:
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model = train_and_evaluate(
            X_train_bat, X_test_bat, y_train, y_test, k_value
        )
        results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx, bat_model]

    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model = train_and_evaluate(
            X_train_cfs, X_test_cfs, y_train, y_test, k_value
        )
        results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx, cfs_model]

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge+delta",
                value=results[method][0],
                title={'text': f"{method} Accuracy"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
        st.plotly_chart(fig)

    # ROC Curve
    if show_roc:
        plt.figure()
        for method in results:
            model = results[method][6]
            y_score = model.predict_proba(X_test_full)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{method} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt)

# ================= Real-Time Prediction ================= #
st.subheader("🔍 Real-Time Heart Disease Prediction")

with st.form("patient_form"):
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
    submit_button = st.form_submit_button("📈 Predict Now")

if submit_button:
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

    input_scaled = scaler.transform(patient_data)
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    model.fit(X_train_full, y_train)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(f"<div style='background-color: #ffcccc; padding: 15px; border-radius: 10px;'><h3 style='color:red;'>⚠️ Positive: Risk of Heart Disease</h3><p>Confidence: {max(proba)*100:.2f}%</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color: #ccffcc; padding: 15px; border-radius: 10px;'><h3 style='color:green;'>✅ Negative: No Heart Disease</h3><p>Confidence: {max(proba)*100:.2f}%</p></div>", unsafe_allow_html=True)
