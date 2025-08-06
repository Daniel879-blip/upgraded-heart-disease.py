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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ================= Feature Selection ================= #
def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8, random_state=42):
    """
    Simple wrapper of your earlier BAT-like selection:
    - binary mask population
    - evaluate using KNN accuracy on held-out split for each bat
    Returns indices of selected features (best bat).
    """
    n_features = X.shape[1]
    rng = np.random.default_rng(random_state)
    population = rng.integers(0, 2, size=(n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            # evaluate on a small validation split
            X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
                X[:, selected], y, test_size=0.2, stratify=y, random_state=42
            )
            # if only one class present after selection, skip
            if len(np.unique(y_train_b)) < 2:
                fitness[i] = 0
                continue
            model_b = KNeighborsClassifier()
            model_b.fit(X_train_b, y_train_b)
            fitness[i] = accuracy_score(y_val_b, model_b.predict(X_val_b))

    best_idx = np.argmax(fitness)
    best_bat = population[best_idx]
    return np.where(best_bat == 1)[0]

def cfs_feature_selection(X_df, y, k=6):
    """
    Correlation-based simple feature ranking (absolute Pearson correlation).
    Returns top-k feature indices (relative to X_df.columns order).
    """
    correlations = []
    for i in range(X_df.shape[1]):
        # Guard against constant column
        if X_df.iloc[:, i].nunique() < 2:
            correlations.append(0.0)
            continue
        correlations.append(abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]))
    correlations = np.nan_to_num(correlations)  # replace any nan with 0
    topk = np.argsort(correlations)[-k:]
    return topk

# ================= Train & Evaluate ================= #
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    """
    Trains KNN on provided X_train/y_train and evaluates on X_test/y_test.
    Returns metrics, confusion matrix, model object, and y_pred.
    """
    # If training set contains only one class, KNN cannot be trained for predict_proba.
    if len(np.unique(y_train)) < 2:
        # Create a dummy classifier via simple constant predictor using majority class
        majority = int(pd.Series(y_train).mode()[0])
        class DummyModel:
            def predict(self, X):
                return np.array([majority] * len(X))
            def predict_proba(self, X):
                # return probabilities: 100% for majority class
                probs = np.zeros((len(X), 2))
                probs[:, majority] = 1.0
                return probs
        model = DummyModel()
        y_pred = model.predict(X_test)
    else:
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm, model, y_pred

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

# Initialize session_state containers for models/indices if not exist
if "models" not in st.session_state:
    st.session_state.models = {}          # store trained model objects per method or 'full'
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = {}    # store indices per method
if "results" not in st.session_state:
    st.session_state.results = {}         # store metrics results per method
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "X_train_full" not in st.session_state:
    st.session_state.X_train_full = None
if "X_test_full" not in st.session_state:
    st.session_state.X_test_full = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

# ================= Load Dataset ================= #
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset")
else:
    df = pd.read_csv("heart.csv")
    st.subheader("📄 Default Dataset Loaded (heart.csv)")

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

# ================= Preview ================= #
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Prepare X and y
X_df = df.drop("target", axis=1)
y = df["target"].values

# Fit scaler on full dataset and save to session_state
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
st.session_state.scaler = scaler

# Create train/test split and store in session_state (so prediction uses same train/test)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)
st.session_state.X_train_full = X_train_full
st.session_state.X_test_full = X_test_full
st.session_state.y_train = y_train
st.session_state.y_test = y_test

# ================= Run Analysis (train once and store models) ================= #
if run_analysis:
    st.session_state.results = {}
    st.session_state.models = {}
    st.session_state.selected_idx = {}

    # BAT branch
    if feature_method in ["BAT", "Both"]:
        # run BAT on training split
        bat_idx = bat_algorithm_feature_selection(X_train_full, y_train, n_bats=12, n_iterations=8)
        # If BAT selects no features, fallback to top-6 by correlation
        if len(bat_idx) == 0:
            st.warning("BAT selected no features; falling back to CFS top-6 features for BAT.")
            bat_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train, k=6)

        X_train_bat, X_test_bat = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model, _ = train_and_evaluate(
            X_train_bat, X_test_bat, y_train, y_test, k_value
        )
        st.session_state.results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]
        st.session_state.models["BAT"] = bat_model
        st.session_state.selected_idx["BAT"] = bat_idx

    # CFS branch
    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train, k=6)
        X_train_cfs, X_test_cfs = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model, _ = train_and_evaluate(
            X_train_cfs, X_test_cfs, y_train, y_test, k_value
        )
        st.session_state.results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]
        st.session_state.models["CFS"] = cfs_model
        st.session_state.selected_idx["CFS"] = cfs_idx

    # Also train a 'full' model on all features as fallback / comparison
    full_acc, full_prec, full_rec, full_f1, full_cm, full_model, _ = train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test, k_value
    )
    st.session_state.results["Full"] = [full_acc, full_prec, full_rec, full_f1, full_cm, np.arange(X_train_full.shape[1])]
    st.session_state.models["Full"] = full_model
    st.session_state.selected_idx["Full"] = np.arange(X_train_full.shape[1])

    st.success("✅ Analysis complete — models trained and saved to session state.")

# If we have results from session_state, use them for display
results = st.session_state.get("results", {})

# ================= Visualizations & Write-ups ================= #
if results:
    st.subheader("📈 Model Comparison & Visualizations")

    # Accuracy Chart
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=results[method][0],
                title={"text": f"{method} Accuracy (%)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
            ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Interpretation:** Accuracy measures how often the classifier correctly predicts heart disease presence or absence.
        Higher accuracy means better model performance.  
        """)

    # Metrics Chart: Precision, Recall, F1
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=metrics, y=results[method][1:4], name=method))
        fig.update_layout(title="Precision / Recall / F1 Score Comparison (%)", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Interpretation:**  
        - **Precision**: Of all predicted positives, how many were correct?  
        - **Recall**: Of all actual positives, how many did we find?  
        - **F1 Score**: Harmonic mean of precision and recall, balancing the two.  
        """)

    # Confusion Matrices
    if show_confusion:
        for method in results:
            st.subheader(f"{method} Confusion Matrix")
            cm = results[method][4]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("""
            **Interpretation:**  
            - **Top-left (TN)**: Correctly predicted no heart disease.  
            - **Top-right (FP)**: Incorrectly predicted heart disease.  
            - **Bottom-left (FN)**: Missed heart disease cases.  
            - **Bottom-right (TP)**: Correctly predicted heart disease.  
            """)

    # Feature importance / selected features
    if show_feature_importance:
        st.subheader("🔎 Selected Features")
        for method in results:
            idx = results[method][5]
            feature_names = X_df.columns[list(idx)]
            st.markdown(f"**{method} selected ({len(feature_names)} features):**  {', '.join(feature_names)}")
        st.markdown("""
        **Interpretation:**  
        - Selected features listed above were chosen by the respective feature selection method.  
        - Compare these to see overlap (consensus) or divergence between BAT and CFS.
        """)

    # ROC Curve: show ROC for each stored model if predict_proba available
    if show_roc_curve:
        fig, ax = plt.subplots()
        plotted_any = False
        for method, model in st.session_state.models.items():
            idx = st.session_state.selected_idx.get(method, np.arange(X_test_full.shape[1]))
            X_test_sub = X_test_full[:, idx]
            # If model has predict_proba, compute ROC; else skip
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_sub)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.2f})')
                plotted_any = True
        if not plotted_any:
            ax.text(0.5, 0.5, "No model supports predict_proba for ROC", horizontalalignment='center')
        ax.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        **Interpretation:**  
        - ROC Curve shows the trade-off between sensitivity (recall) and specificity.  
        - AUC closer to **1.0** indicates a better model.  
        """)

    # Distribution Plots
    if show_distribution_plots:
        st.subheader("📊 Feature Distributions")
        if 'age' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df, x='age', hue='target', multiple='stack', ax=ax)
            ax.set_title("Age distribution by target")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("""
            **Interpretation:**  
            - Shows the age distribution of patients by heart disease status.  
            - Helps identify age groups with higher heart disease prevalence.  
            """)
        else:
            st.info("No 'age' column found to plot distribution.")

    # Pair Plot
    if show_pairplot:
        st.markdown("📊 **Pair Plot for Feature Relationships**")
        subset_cols = [c for c in ['age', 'chol', 'thalach', 'target'] if c in df.columns]
        if len(subset_cols) >= 2:
            pair_fig = sns.pairplot(df[subset_cols], hue='target').fig
            st.pyplot(pair_fig)
            plt.close(pair_fig)
            st.markdown("""
            **Interpretation:**  
            - Each point represents a patient.  
            - Diagonal = distribution of each feature.  
            - Off-diagonals = correlation between features.  
            """)
        else:
            st.info("Not enough columns available to plot pairplot (needs at least 2 features + target).")

# ================= Real-Time Prediction ================= #
st.subheader("🔍 Real-Time Heart Disease Prediction")
st.markdown("Enter patient details to predict heart disease risk. The app will use the model you trained in the analysis run (BAT or CFS or Full). If you haven't run analysis, predictions will use a model trained on the full dataset as fallback.")

with st.form("patient_form"):
    # Controls — keep exactly as your original fields
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, format="%.1f")
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    submit_button = st.form_submit_button("📈 Predict Now")

if submit_button:
    # maps (same as your original)
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    # Make sure columns order matches X_df.columns
    # If the uploaded dataset uses different encodings for categorical columns,
    # user must ensure columns match expected order. We'll assume standard heart.csv ordering.
    try:
        patient_data = pd.DataFrame([[
            age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs],
            restecg_map[restecg], thalach, exang_map[exang], oldpeak,
            slope_map[slope], ca, thal_map[thal]
        ]], columns=X_df.columns)
    except Exception as e:
        st.error(f"Error creating input row: {e}")
        st.stop()

    # scale using stored scaler
    scaler = st.session_state.scaler
    input_scaled = scaler.transform(patient_data)

    # Choose which model to use for prediction:
    # Priority: If analysis ran and BAT exists, use BAT if feature_method selected BAT or Both.
    model_to_use = None
    selected_idx = None

    # If analysis run and BAT present and user selected BAT in sidebar or Both, prefer BAT
    if feature_method in ["BAT", "Both"] and "BAT" in st.session_state.models:
        model_to_use = st.session_state.models["BAT"]
        selected_idx = st.session_state.selected_idx["BAT"]
    elif feature_method in ["CFS", "Both"] and "CFS" in st.session_state.models:
        model_to_use = st.session_state.models["CFS"]
        selected_idx = st.session_state.selected_idx["CFS"]
    else:
        # fallback to full model
        model_to_use = st.session_state.models.get("Full", None)
        selected_idx = st.session_state.selected_idx.get("Full", np.arange(input_scaled.shape[1]))

    if model_to_use is None:
        st.error("No trained model available. Please run 'Train Model & Compare' first.")
    else:
        # Apply same selected indices as the model expects
        input_sub = input_scaled[:, selected_idx]

        # If model doesn't have predict_proba guarantee it by fallback
        if not hasattr(model_to_use, "predict_proba"):
            # create simple wrapper that conforms
            # This occurs if training had only one class in y_train and DummyModel used.
            # DummyModel has predict_proba in our implementation, but we keep guard.
            class Wrapped:
                def __init__(self, m):
                    self.m = m
                def predict(self, X):
                    return self.m.predict(X)
                def predict_proba(self, X):
                    probs = np.zeros((len(X), 2))
                    preds = self.m.predict(X)
                    for i, p in enumerate(preds):
                        probs[i, p] = 1.0
                    return probs
            model_to_use = Wrapped(model_to_use)

        prediction = model_to_use.predict(input_sub)[0]
        proba = model_to_use.predict_proba(input_sub)[0]

        # display
        if prediction == 1:
            st.error(f"🛑 Positive (Heart Disease) — Confidence: {max(proba)*100:.2f}%")
        else:
            st.success(f"✅ Negative (No Heart Disease) — Confidence: {max(proba)*100:.2f}%")

        # show which model & features used
        used_features = list(X_df.columns[selected_idx])
        st.markdown(f"**Model used:** `{[k for k,v in st.session_state.models.items() if v is model_to_use or st.session_state.models.get('Full') is model_to_use][0] if model_to_use else 'Unknown'}`")
        st.markdown(f"**Features used ({len(used_features)}):** {', '.join(used_features)}")

# ================= Footer Write-up (your explanations preserved) ================= #
st.markdown("---")
st.markdown("""
## Notes & Interpretation

- **Why we save models in session state:**  
  The real-time prediction uses the *exact same model object* trained during the analysis phase (BAT or CFS or Full). This avoids retraining differences and scaling mismatches which can cause predictions to bias toward one class.

- **Why KNN can predict the majority class often:**  
  - KNN is distance-based and depends strongly on feature scaling and the neighborhood (k).  
  - If the dataset is imbalanced (more negatives than positives), KNN may default to the majority unless `weights='distance'` and careful feature selection helps.

- **Tips:**  
  - Run the analysis (`🚀 Train Model & Compare`) before trying live predictions.  
  - Try different k values, or try stratified resampling if your dataset is highly imbalanced.  
  - Consider other classifiers (RandomForest, LogisticRegression) for more stable probability estimates.

**If you want:** I can:
- Add an explicit dropdown to let you pick **which trained model** to use for prediction (BAT/CFS/Full).
- Add class imbalance handling (SMOTE/oversampling) in training.
- Add saving/loading of trained models to disk for persistent use.

""")
