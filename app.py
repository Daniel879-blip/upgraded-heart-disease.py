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

# ================= Feature Selection ================= #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    n_features = X.shape[1]
    rng = np.random.default_rng(42)
    population = rng.integers(0, 2, size=(n_bats, n_features))
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
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        confusion_matrix(y_test, y_pred),
        model,
        y_pred
    )

# ================= Streamlit Setup ================= #
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")
st.sidebar.title("⚙️ Settings & Controls")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection Method", ["Both", "BAT", "CFS"])

# NEW: Classifier choice (currently only KNN)
classifier_choice = st.sidebar.selectbox("🤖 Classifier", ["KNN"])
k_value = st.sidebar.slider("🔢 K Value for KNN", 1, 15, 7)

test_size = st.sidebar.slider("📊 Test Size (%)", 10, 50, 20, step=5) / 100
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_feature_importance = st.sidebar.checkbox("🏅 Show Feature Importance", True)
show_roc_curve = st.sidebar.checkbox("📊 Show ROC Curve", True)
show_distribution_plots = st.sidebar.checkbox("📊 Show Feature Distributions", True)
show_pairplot = st.sidebar.checkbox("🔗 Show Pair Plot", True)
run_analysis = st.sidebar.button("🚀 Train Model & Compare")

# ================= Load Dataset ================= #
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")
    st.subheader("📄 Default Dataset Loaded (heart.csv)")

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# ================= Preview ================= #
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y, random_state=42)

# ================= Run Analysis ================= #
if run_analysis:
    results = {}
    
    if feature_method in ["BAT", "Both"]:
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, _, _ = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test, k_value)
        results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]
    
    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, _, _ = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test, k_value)
        results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=results[method][0],
                title={"text": f"{method} Accuracy"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
            ))
        st.plotly_chart(fig)

    # Metrics Chart
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=metrics, y=results[method][1:4], name=method))
        st.plotly_chart(fig)

    # Confusion Matrices
    if show_confusion:
        for method in results:
            st.subheader(f"{method} Confusion Matrix")
            sns.heatmap(results[method][4], annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt.gcf())

    # ROC Curve
    if show_roc_curve:
        fig, ax = plt.subplots()
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train_full, y_train)
        y_proba = model.predict_proba(X_test_full)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.legend()
        st.pyplot(fig)

# ================= Real-Time Prediction ================= #
st.subheader("🔍 Real-Time Heart Disease Prediction")

with st.form("patient_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
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

    # FIX: Prediction shows both positive & negative correctly
    if prediction == 1:
        st.error(f"🛑 Positive (Heart Disease) — Confidence: {max(proba)*100:.2f}%")
    else:
        st.success(f"✅ Negative (No Heart Disease) — Confidence: {max(proba)*100:.2f}%")
