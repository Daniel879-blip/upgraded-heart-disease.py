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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- Feature Selection ---------------- #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    """
    Binary BAT Algorithm for feature selection.
    Selects subset of features that maximizes KNN classifier accuracy.
    """
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

def cfs_feature_selection(X_df, y, k=6):
    """
    Correlation-based Feature Selection.
    Selects top k features most correlated with the target variable.
    """
    correlations = [
        abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1])
        for i in range(X_df.shape[1])
    ]
    return np.argsort(correlations)[-k:]

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
st.set_page_config(page_title="BAT vs CFS on KNN", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 App Navigation")
page = st.sidebar.radio("Go to:", ["Dataset Overview", "Comparative Analysis", "Real-Time Prediction"])

# Sidebar Dataset Upload
st.sidebar.subheader("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded!")
else:
    try:
        df = pd.read_csv("heart.csv")
        st.sidebar.info("ℹ️ Using default heart.csv dataset")
    except FileNotFoundError:
        st.error("❌ No dataset found. Please upload one.")
        st.stop()

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# Sidebar Info Panel
st.sidebar.subheader("ℹ️ About Feature Selection")
st.sidebar.write("**BAT**: Bio-inspired algorithm that optimizes feature selection by mimicking bat echolocation.")
st.sidebar.write("**CFS**: Selects features most correlated with the target variable, removing irrelevant ones.")
st.sidebar.write("**KNN**: Classifies new data points based on the majority class of their nearest neighbors.")

# Prepare Data
X_df = df.drop("target", axis=1)
y = df["target"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Dataset Overview Page ---------------- #
if page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("""
    This section gives you a first look at the dataset before we apply any machine learning.
    We inspect the target distribution, feature relationships, and correlations.
    """)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())
    st.markdown("**Explanation:** This table shows the first 5 rows of the dataset including patient details and the target column.")

    st.subheader("📈 Target Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="target", palette="Set2", ax=ax)
    st.pyplot(fig)
    st.markdown("**Interpretation:** '0' means no heart disease, '1' means presence of heart disease.")

    st.subheader("🔗 Feature Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    st.pyplot(plt)
    st.markdown("**Interpretation:** Brighter colors indicate stronger correlations between features and the target.")

# ---------------- Comparative Analysis Page ---------------- #
elif page == "Comparative Analysis":
    st.title("⚖️ Comparative Analysis: BAT vs CFS with KNN")
    st.write("""
    **BAT** is a metaheuristic optimization algorithm inspired by bat echolocation, searching for the best subset of features.  
    **CFS** selects features with the highest correlation to the target variable, reducing redundancy.  
    Both are tested with **KNN** to see which produces better prediction performance.
    """)

    if st.button("🚀 Run Analysis"):
        st.info("Step 1: Running BAT Feature Selection...")
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm = train_and_evaluate(X_train_bat, X_test_bat, y_train, y_test)

        st.info("Step 2: Running CFS Feature Selection...")
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm = train_and_evaluate(X_train_cfs, X_test_cfs, y_train, y_test)

        st.success("✅ Analysis Completed!")

        st.subheader("🔍 Feature Selection Comparison")
        venn2([set(X_df.columns[bat_idx]), set(X_df.columns[cfs_idx])],
              set_labels=('BAT Selected Features', 'CFS Selected Features'))
        st.pyplot(plt)
        st.markdown("**Interpretation:** Overlapping area shows features chosen by both BAT and CFS.")

        st.subheader("📊 Performance Table")
        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
            "BAT": [bat_acc, bat_prec, bat_rec, bat_f1],
            "CFS": [cfs_acc, cfs_prec, cfs_rec, cfs_f1]
        })
        st.dataframe(comparison_df)
        st.markdown("**Interpretation:** Higher numbers indicate better performance. Accuracy is the main metric here.")

        st.subheader("📌 Confusion Matrices")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BAT Confusion Matrix**")
            sns.heatmap(bat_cm, annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt.gcf())
        with col2:
            st.markdown("**CFS Confusion Matrix**")
            sns.heatmap(cfs_cm, annot=True, fmt="d", cmap="Greens")
            st.pyplot(plt.gcf())

# ---------------- Real-Time Prediction Page ---------------- #
elif page == "Real-Time Prediction":
    st.title("🔍 Real-Time Heart Disease Prediction")
    st.write("""
    Enter patient details below.  
    The model will use the trained **KNN** classifier with the selected features to predict if the patient is at risk.
    """)

    input_data = {col: st.number_input(f"{col}", format="%.2f") for col in X_df.columns}
    if st.button("📈 Predict Now"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        model = KNeighborsClassifier(n_neighbors=7, weights='distance')
        model.fit(X_train_full, y_train)
        prediction = model.predict(input_scaled)[0]
        result = "Positive (Risk of Heart Disease)" if prediction == 1 else "Negative (No Risk)"
        st.success(f"Prediction: {result}")
        st.markdown("""
        **Interpretation:**  
        - **Positive:** Patient likely has heart disease.  
        - **Negative:** Patient unlikely to have heart disease.  
        This is based on similarity to previous patients in the dataset.
        """)
