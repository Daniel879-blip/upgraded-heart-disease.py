import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ---------------- OPTIMIZED BAT Feature Selection ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=10):
    """Optimized BAT algorithm for feature selection."""
    n_features = X.shape[1]
    rng = np.random.default_rng()
    population = rng.integers(0, 2, size=(n_bats, n_features))
    velocities = np.zeros((n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            selected = rng.integers(0, n_features, size=1)
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

# ---------------- OPTIMIZED CFS Feature Selection ---------------- #
def cfs_feature_selection(X_df, y, k=6):
    """Correlation-based Feature Selection."""
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1])
        for i in range(X_df.shape[1])
    ]
    return np.argsort(correlations)[-k:]

# ---------------- Train and Evaluate KNN ---------------- #
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate KNN."""
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
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
page = st.sidebar.radio("Go to:", [
    "Dataset Overview", "Comparative Analysis", "Performance Analysis", "Real-Time Prediction", "Risk Report"
])

# Sidebar Upload Dataset
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("heart.csv")  # Default dataset
    except FileNotFoundError:
        st.error("❌ No dataset found. Please upload one.")
        st.stop()

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Prepare Data
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Dataset Overview ---------------- #
if page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.subheader("Dataset Summary")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(df.head())

    st.subheader("📈 Target Class Distribution")
    fig1, ax1 = plt.subplots()
    df['target'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90, ax=ax1
    )
    ax1.set_ylabel('')
    st.pyplot(fig1)

    st.subheader("🔗 Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------------- Comparative Analysis ---------------- #
elif page == "Comparative Analysis":
    st.title("⚖️ Comparative Analysis: BAT vs CFS using KNN")
    if st.button("🚀 Run Analysis"):
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, _ = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test)

        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, _ = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test)

        venn2([set(X_df.columns[bat_idx]), set(X_df.columns[cfs_idx])],
              set_labels=('BAT Features', 'CFS Features'))
        st.pyplot(plt)

        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "BAT": [bat_acc, bat_prec, bat_rec, bat_f1],
            "CFS": [cfs_acc, cfs_prec, cfs_rec, cfs_f1]
        })
        st.dataframe(comparison_df)

# ---------------- Performance Analysis ---------------- #
elif page == "Performance Analysis":
    st.title("📊 Performance Analysis")
    st.write("Performance comparison charts and ROC curves will be here.")

# ---------------- Real-Time Prediction ---------------- #
elif page == "Real-Time Prediction":
    st.title("🔍 Real-Time Heart Disease Prediction (KNN)")
    input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
    if st.button("📈 Predict Now"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        model = KNeighborsClassifier(n_neighbors=7, weights='distance')
        model.fit(X_train_full, y_train)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {max(proba)*100:.2f}%")

# ---------------- Risk Report ---------------- #
elif page == "Risk Report":
    st.title("📑 Patient Risk Report")
    st.write("This section will show a detailed breakdown of patient predictions.")
