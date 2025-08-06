import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- BAT Feature Selection ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    """BAT algorithm for feature selection."""
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
                X[:, selected], y, test_size=0.3, stratify=y
            )
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
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.3, stratify=y
            )
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

# ---------------- CFS Feature Selection ---------------- #
def cfs_feature_selection(X_df, y, k=6):
    """Correlation-based Feature Selection."""
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1])
        for i in range(X_df.shape[1])
    ]
    return np.argsort(correlations)[-k:]

# ---------------- Train and Evaluate KNN ---------------- #
def train_and_evaluate(X_train, X_test, y_train, y_test, k_neighbors=7):
    """Train and evaluate KNN."""
    model = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        confusion_matrix(y_test, y_pred),
        model
    )

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Dataset Overview", "Comparative Analysis", "Real-Time Prediction"])

# Sidebar Settings
feature_choice = st.sidebar.selectbox("🧠 Feature Selection Method", ["BAT", "CFS", "Both"])
k_neighbors = st.sidebar.slider("🔢 K Value (KNN)", 1, 20, 7)
test_size = st.sidebar.slider("📊 Test Size (%)", 10, 50, 20) / 100
metric_style = st.sidebar.radio("📏 Metric Display Style", ["Percentage", "Decimal"])
show_patient_info = st.sidebar.checkbox("🩺 Show Patient Information Form", True)
show_venn = st.sidebar.checkbox("📊 Show Venn Diagram", True)
show_perf_table = st.sidebar.checkbox("📊 Show Performance Table", True)
show_conf_matrices = st.sidebar.checkbox("📊 Show Confusion Matrices", True)

# Dataset Upload
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")  # Default dataset

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Prepare Data
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# ---------------- Dataset Overview Page ---------------- #
if page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("This section gives you a complete understanding of the dataset before applying feature selection and KNN.")

    # Dataset Summary
    st.subheader("📄 Dataset Summary")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Class Distribution:**\n{df['target'].value_counts(normalize=True)*100}")

    # Pie Chart
    st.subheader("📈 Target Class Distribution")
    fig1, ax1 = plt.subplots()
    df['target'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90, ax=ax1
    )
    ax1.set_ylabel('')
    st.pyplot(fig1)
    st.markdown("**Interpretation:** '0' = No heart disease, '1' = Presence of heart disease.")

    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    st.markdown("**Interpretation:** Brighter colors indicate stronger correlations with the target variable.")

# ---------------- Comparative Analysis Page ---------------- #
elif page == "Comparative Analysis":
    st.title("⚖️ Comparative Analysis: BAT vs CFS using KNN")
    if st.button("🚀 Run Analysis"):
        results = {}

        if feature_choice in ["BAT", "Both"]:
            bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
            X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
            bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, _ = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test, k_neighbors)
            results["BAT"] = (bat_acc, bat_prec, bat_rec, bat_f1, bat_cm)

        if feature_choice in ["CFS", "Both"]:
            cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
            X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
            cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, _ = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test, k_neighbors)
            results["CFS"] = (cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm)

        # Venn Diagram
        if show_venn and feature_choice == "Both":
            st.subheader("🔍 Feature Selection Comparison")
            venn2([set(X_df.columns[bat_idx]), set(X_df.columns[cfs_idx])],
                  set_labels=('BAT Features', 'CFS Features'))
            st.pyplot(plt)
            st.markdown("**Interpretation:** Overlap = features selected by both methods.")

        # Performance Table
        if show_perf_table and len(results) == 2:
            st.subheader("📊 Performance Comparison")
            comparison_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "BAT": [results["BAT"][0], results["BAT"][1], results["BAT"][2], results["BAT"][3]],
                "CFS": [results["CFS"][0], results["CFS"][1], results["CFS"][2], results["CFS"][3]]
            })

            if metric_style == "Percentage":
                comparison_df.iloc[:, 1:] = comparison_df.iloc[:, 1:] * 100

            st.dataframe(comparison_df)

        # Confusion Matrices
        if show_conf_matrices:
            st.subheader("📌 Confusion Matrices")
            for method, values in results.items():
                st.markdown(f"**{method} Confusion Matrix**")
                sns.heatmap(values[4], annot=True, fmt="d", cmap="Blues")
                st.pyplot(plt.gcf())

# ---------------- Real-Time Prediction Page ---------------- #
elif page == "Real-Time Prediction":
    st.title("🔍 Real-Time Heart Disease Prediction (KNN)")

    if show_patient_info:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol Level", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels", 0, 4, 0)
        thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

        # Convert categorical to numerical
        input_data = [
            age, 1 if sex == "Male" else 0, ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp),
            trestbps, chol, 1 if fbs == "Yes" else 0, ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            thalach, 1 if exang == "Yes" else 0, oldpeak, ["Upsloping", "Flat", "Downsloping"].index(slope),
            ca, ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        ]

    else:
        input_data = [st.number_input(f"{col}", format="%.2f") for col in X_df.columns]

    if st.button("📈 Predict Now"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        model = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
        model.fit(X_train_full, y_train)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {max(proba)*100:.2f}%")
