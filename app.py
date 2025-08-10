import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

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
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# ================= Run Analysis ================= #
if run_analysis:
    st.session_state.results = {}
    st.session_state.selected_features = {}
    st.session_state.models = {}

    # BAT Feature Selection & Training
    if feature_method in ["BAT", "Both"]:
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train)
        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model, _ = train_and_evaluate(
            X_train_bat, X_test_bat, y_train, y_test, k_value
        )
        st.session_state.results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]
        st.session_state.selected_features["BAT"] = bat_idx
        st.session_state.models["BAT"] = bat_model

    # CFS Feature Selection & Training
    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model, _ = train_and_evaluate(
            X_train_cfs, X_test_cfs, y_train, y_test, k_value
        )
        st.session_state.results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]
        st.session_state.selected_features["CFS"] = cfs_idx
        st.session_state.models["CFS"] = cfs_model

    # ================= Visualization & Interpretation ================= #

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in st.session_state.results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=st.session_state.results[method][0],
                title={"text": f"{method} Accuracy"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
            ))
        st.plotly_chart(fig)
        st.markdown("""
        **Interpretation:** Accuracy measures how often the classifier correctly predicts heart disease presence or absence.  
        Higher accuracy means better model performance.  
        """)

    # Metrics Chart (Precision, Recall, F1)
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        for method in st.session_state.results:
            fig.add_trace(go.Bar(x=metrics, y=st.session_state.results[method][1:4], name=method))
        fig.update_layout(title="Precision / Recall / F1 Score Comparison (%)")
        st.plotly_chart(fig)
        st.markdown("""
        **Interpretation:**  
        - **Precision**: Of all predicted positives, how many were correct?  
        - **Recall**: Of all actual positives, how many did we find?  
        - **F1 Score**: Harmonic mean of precision and recall, balancing the two.  
        """)

    # Confusion Matrices
    if show_confusion:
        for method in st.session_state.results:
            st.subheader(f"{method} Confusion Matrix")
            sns.heatmap(
                st.session_state.results[method][4], annot=True, fmt="d",
                cmap="Blues", cbar=False
            )
            st.pyplot(plt.gcf())
            st.markdown("""
            **Interpretation:**  
            - **Top-left (TN)**: Correctly predicted no heart disease.  
            - **Top-right (FP)**: Incorrectly predicted heart disease.  
            - **Bottom-left (FN)**: Missed heart disease cases.  
            - **Bottom-right (TP)**: Correctly predicted heart disease.  
            """)

    # Feature Importance (based on selection frequency)
    if show_feature_importance:
        st.subheader("🏅 Feature Importance (Selected Features)")
        for method in st.session_state.selected_features:
            feature_idx = st.session_state.selected_features[method]
            feature_names = X_df.columns[feature_idx].tolist()
            st.write(f"**{method} Selected Features:** {', '.join(feature_names)}")

    # ROC Curve (Using full dataset and KNN)
    if show_roc_curve:
        fig, ax = plt.subplots()
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train_full, y_train)
        y_proba = model.predict_proba(X_test_full)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.markdown("""
        **Interpretation:**  
        ROC curve shows the trade-off between sensitivity and specificity.  
        The closer the curve is to the top-left, the better the model.  
        """)

    # Distribution Plots
    if show_distribution_plots:
        st.subheader("📊 Feature Distributions")
        for col in X_df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, hue="target", multiple="stack", ax=ax)
            plt.title(f"Distribution of {col} by Target Class")
            st.pyplot(fig)

    # Pairplot
    if show_pairplot:
        st.subheader("🔗 Pair Plot of Features")
        pairplot_fig = sns.pairplot(df, hue="target", diag_kind="kde")
        st.pyplot(pairplot_fig)

else:
    st.info("➡️ Upload your dataset and click 'Train Model & Compare' in the sidebar to begin.")

# ================= Real-Time Prediction ================= #
st.sidebar.markdown("---")
st.sidebar.header("🔮 Real-Time Prediction")
uploaded_single = st.sidebar.file_uploader("Upload single sample CSV (same features)", type=["csv"], key="single")
if uploaded_single and run_analysis:
    sample_df = pd.read_csv(uploaded_single)
    if list(sample_df.columns) != list(X_df.columns):
        st.sidebar.error("❌ Features mismatch with training data.")
    else:
        sample_scaled = scaler.transform(sample_df)
        pred_methods = list(st.session_state.models.keys())
        selected_method = st.sidebar.selectbox("Choose Model for Prediction", pred_methods)

        model = st.session_state.models[selected_method]

        if selected_method == "BAT":
            selected_idx = st.session_state.selected_features["BAT"]
            sample_scaled = sample_scaled[:, selected_idx]
        elif selected_method == "CFS":
            selected_idx = st.session_state.selected_features["CFS"]
            sample_scaled = sample_scaled[:, selected_idx]

        prediction = model.predict(sample_scaled)
        prediction_proba = model.predict_proba(sample_scaled)[:, 1]

        st.sidebar.success(f"Prediction: {'Heart Disease' if prediction[0]==1 else 'No Heart Disease'}")
        st.sidebar.info(f"Prediction Probability: {prediction_proba[0]*100:.2f}%")
