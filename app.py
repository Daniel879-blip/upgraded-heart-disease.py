# save as app.py and run: streamlit run app.py

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

# ------------------- Improved BAT (random start + bit-flip hill-climb) -------------------
def bat_algorithm_feature_selection(X, y, n_starts=8, n_iters=50, random_state=42):
    """
    Simple randomized local search for binary feature masks:
    - multiple random starts
    - at each iteration flip one bit and accept if accuracy on small validation improves
    Returns best feature indices found.
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    best_mask = None
    best_score = -np.inf

    # Use a single small validation split for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # If there is only one class in y_tr, return all features to avoid degenerate behavior
    if len(np.unique(y_tr)) < 2:
        return np.arange(n_features)

    for start in range(n_starts):
        # random initial mask (at least one feature)
        mask = rng.integers(0,2, size=n_features)
        if mask.sum() == 0:
            mask[rng.integers(0,n_features)] = 1

        # evaluate initial
        sel = np.where(mask==1)[0]
        model = KNeighborsClassifier()
        try:
            model.fit(X_tr[:, sel], y_tr)
            current_score = accuracy_score(y_val, model.predict(X_val[:, sel]))
        except Exception:
            current_score = -np.inf

        for it in range(n_iters):
            # flip a random bit
            i = rng.integers(0, n_features)
            new_mask = mask.copy()
            new_mask[i] = 1 - new_mask[i]
            # ensure at least one feature
            if new_mask.sum() == 0:
                continue
            sel_new = np.where(new_mask==1)[0]
            try:
                model_new = KNeighborsClassifier()
                model_new.fit(X_tr[:, sel_new], y_tr)
                new_score = accuracy_score(y_val, model_new.predict(X_val[:, sel_new]))
            except Exception:
                new_score = -np.inf

            # accept if better
            if new_score >= current_score:
                mask = new_mask
                current_score = new_score

        # after iterations, compare to global best
        if current_score > best_score:
            best_score = current_score
            best_mask = mask.copy()

    if best_mask is None:
        # fallback: choose top feature (if something went wrong)
        return np.array([0])
    return np.where(best_mask == 1)[0]

# ------------------- CFS (simple absolute Pearson ranking) -------------------
def cfs_feature_selection(X_df, y, k=6):
    correlations = []
    for i in range(X_df.shape[1]):
        col = X_df.iloc[:, i]
        if col.nunique() < 2:
            correlations.append(0.0)
            continue
        corr = abs(np.corrcoef(col, y)[0,1])
        if np.isnan(corr):
            corr = 0.0
        correlations.append(corr)
    correlations = np.array(correlations)
    topk = np.argsort(correlations)[-k:]
    return topk

# ------------------- Train & evaluate helper -------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    # handle degenerate single-class training
    if len(np.unique(y_train)) < 2:
        majority = int(pd.Series(y_train).mode()[0])
        class DummyModel:
            def predict(self, X):
                return np.array([majority] * len(X))
            def predict_proba(self, X):
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

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="BAT vs CFS on KNN (Fixed)", layout="wide")
st.title("BAT vs CFS on KNN — Debugged & Improved")

st.sidebar.title("⚙️ Settings & Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
feature_method = st.sidebar.selectbox("Feature Selection Method", ["Both", "BAT", "CFS"])
k_value = st.sidebar.slider("K for KNN", 1, 15, 7)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100
run_analysis = st.sidebar.button("Train & Compare")
show_diagnostics = st.sidebar.checkbox("Show Diagnostics (helpful)", True)

# options for charts
show_accuracy_chart = st.sidebar.checkbox("Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("Show Metrics Chart", True)
show_confusion = st.sidebar.checkbox("Show Confusion Matrices", True)
show_roc_curve = st.sidebar.checkbox("Show ROC Curve", True)
show_distribution_plots = st.sidebar.checkbox("Show Distribution Plots", True)
show_pairplot = st.sidebar.checkbox("Show Pairplot", True)

# Load dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded dataset")
else:
    df = pd.read_csv("heart.csv")
    st.subheader("Default dataset (heart.csv)")

if "target" not in df.columns:
    st.error("Dataset must contain 'target' column.")
    st.stop()

st.dataframe(df.head())

# Prepare X,y and scaler
X_df = df.drop("target", axis=1)
y = df["target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# session state to hold models and selections
if "models" not in st.session_state:
    st.session_state.models = {}
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = {}
if "results" not in st.session_state:
    st.session_state.results = {}

# Diagnostics: class balance
if show_diagnostics:
    st.markdown("### Dataset Diagnostics")
    class_counts = pd.Series(y).value_counts().to_dict()
    st.write("Target distribution (full dataset):", class_counts)
    st.write("Train set distribution:", pd.Series(y_train).value_counts().to_dict())
    st.write("Test set distribution:", pd.Series(y_test).value_counts().to_dict())

# Run analysis
if run_analysis:
    st.session_state.models = {}
    st.session_state.selected_idx = {}
    st.session_state.results = {}

    # BAT
    if feature_method in ["BAT", "Both"]:
        with st.spinner("Running BAT feature search..."):
            bat_idx = bat_algorithm_feature_selection(X_train_full, y_train, n_starts=12, n_iters=80)
            if len(bat_idx) == 0:
                st.warning("BAT found no features; falling back to top-6 CFS.")
                bat_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train, k=6)
            X_tr_b, X_te_b = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
            bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model, bat_y_pred = train_and_evaluate(
                X_tr_b, X_te_b, y_train, y_test, k_value
            )
            st.session_state.models["BAT"] = bat_model
            st.session_state.selected_idx["BAT"] = bat_idx
            st.session_state.results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx]

    # CFS (compute correlations using original unscaled dataframe for interpretability)
    if feature_method in ["CFS", "Both"]:
        with st.spinner("Running CFS selection..."):
            cfs_idx = cfs_feature_selection(X_df, y, k=min(6, X_df.shape[1]))
            X_tr_c, X_te_c = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
            cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model, cfs_y_pred = train_and_evaluate(
                X_tr_c, X_te_c, y_train, y_test, k_value
            )
            st.session_state.models["CFS"] = cfs_model
            st.session_state.selected_idx["CFS"] = cfs_idx
            st.session_state.results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx]

    # Full model baseline
    X_tr_f, X_te_f = X_train_full, X_test_full
    full_acc, full_prec, full_rec, full_f1, full_cm, full_model, full_y_pred = train_and_evaluate(
        X_tr_f, X_te_f, y_train, y_test, k_value
    )
    st.session_state.models["Full"] = full_model
    st.session_state.selected_idx["Full"] = np.arange(X_train_full.shape[1])
    st.session_state.results["Full"] = [full_acc, full_prec, full_rec, full_f1, full_cm, np.arange(X_train_full.shape[1])]

    st.success("Training complete — models stored in session state.")

# Show results if present
results = st.session_state.get("results", {})
if results:
    st.subheader("Model comparison")
    # Show selected features for each model
    for method in results:
        idx = results[method][5]
        feature_names = list(X_df.columns[list(idx)])
        st.markdown(f"**{method} selected ({len(feature_names)} features):** {', '.join(feature_names)}")

    # Accuracy gauge
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=results[method][0],
                title={"text": f"{method} Accuracy (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ))
        st.plotly_chart(fig, use_container_width=True)

    # Metrics bar chart
    if show_metrics_chart:
        metrics = ["Precision", "Recall", "F1"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=metrics, y=results[method][1:4], name=method))
        fig.update_layout(barmode='group', title="Precision / Recall / F1 (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices + prediction distributions
    if show_confusion:
        for method in results:
            st.subheader(f"{method} confusion matrix")
            cm = results[method][4]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
            # Also show distribution of predictions vs true labels on test set
            model = st.session_state.models.get(method)
            idx = st.session_state.selected_idx.get(method)
            X_test_sub = X_test_full[:, idx]
            preds = model.predict(X_test_sub)
            st.write(f"{method} test predictions distribution:", pd.Series(preds).value_counts().to_dict())
            st.write(f"{method} test true distribution:", pd.Series(y_test).value_counts().to_dict())

    # ROC
    if show_roc_curve:
        fig, ax = plt.subplots()
        any_plotted = False
        for method, model in st.session_state.models.items():
            idx = st.session_state.selected_idx[method]
            X_test_sub = X_test_full[:, idx]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_sub)[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{method} (AUC={roc_auc:.2f})")
                any_plotted = True
        if not any_plotted:
            ax.text(0.5, 0.5, "No model supports predict_proba", ha='center')
        ax.plot([0,1],[0,1], linestyle='--', color='red')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# Distribution and pairplot (keep original write-ups)
if show_distribution_plots:
    if 'age' in df.columns:
        st.subheader("Age distribution by target")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df, x='age', hue='target', multiple='stack', ax=ax)
        st.pyplot(fig)
        plt.close(fig)

if show_pairplot:
    subset_cols = [c for c in ['age','chol','thalach','target'] if c in df.columns]
    if len(subset_cols) >= 2:
        st.subheader("Pair plot")
        pair_fig = sns.pairplot(df[subset_cols], hue='target').fig
        st.pyplot(pair_fig)
        plt.close(pair_fig)

# ------------------- Real-time prediction UI -------------------
st.subheader("Real-time prediction")

with st.form("patient_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male","Female"])
    cp = st.selectbox("Chest Pain", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting BS >120", ["Yes","No"])
    restecg = st.selectbox("Resting ECG", ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max HR", 60, 220, 150)
    exang = st.selectbox("Exercise induced angina", ["Yes","No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping","Flat","Downsloping"])
    ca = st.number_input("Number of major vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversible Defect"])
    chosen_model_label = st.selectbox("Model to use for prediction", options=list(st.session_state.models.keys())+["Full"] if st.session_state.models else ["Full"])
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # mapping
    sex_map = {"Male":1,"Female":0}
    cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
    fbs_map = {"Yes":1,"No":0}
    restecg_map = {"Normal":0,"ST-T Wave Abnormality":1,"Left Ventricular Hypertrophy":2}
    exang_map = {"Yes":1,"No":0}
    slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
    thal_map = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}

    patient_data = pd.DataFrame([[
        age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs],
        restecg_map[restecg], thalach, exang_map[exang], oldpeak,
        slope_map[slope], ca, thal_map[thal]
    ]], columns=X_df.columns)

    input_scaled = scaler.transform(patient_data)

    # choose model
    model_key = chosen_model_label if chosen_model_label in st.session_state.models else "Full"
    model_used = st.session_state.models.get(model_key, st.session_state.models.get("Full"))
    if model_used is None:
        st.error("No model available — run Train & Compare first.")
    else:
        selected_idx = st.session_state.selected_idx.get(model_key, np.arange(X_train_full.shape[1]))
        input_sub = input_scaled[:, selected_idx]

        # ensure predict_proba exists
        if not hasattr(model_used, "predict_proba"):
            class Wrapped:
                def __init__(self, m):
                    self.m = m
                def predict(self, X):
                    return self.m.predict(X)
                def predict_proba(self, X):
                    preds = self.m.predict(X)
                    probs = np.zeros((len(X),2))
                    for i,p in enumerate(preds):
                        probs[i,p] = 1.0
                    return probs
            model_used = Wrapped(model_used)

        pred = model_used.predict(input_sub)[0]
        proba = model_used.predict_proba(input_sub)[0]
        if pred == 1:
            st.error(f"Positive (Heart Disease). Confidence: {max(proba)*100:.2f}%")
        else:
            st.success(f"Negative (No Heart Disease). Confidence: {max(proba)*100:.2f}%")

        st.write("Model used:", model_key)
        st.write("Features used:", list(X_df.columns[selected_idx]))

# final notes
st.markdown("---")
st.markdown("""
**Notes**

- If a model predicts only the majority class on the test set, you'll see that in the "test predictions distribution" diagnostics printed above. If you see `{1:0}` or `{0:some_large}`, that's the issue — the model is always predicting one class.
- Solutions: try smaller k (e.g., 1 or 3), resample the training set (SMOTE/oversample), or use a different classifier (RandomForest, LogisticRegression).
- If BAT still returns very similar features to CFS, increase `n_starts`/`n_iters` or try seeding with different random_state to explore different parts of the search space.
""")
