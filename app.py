import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

st.set_page_config(page_title="BAT vs CFS on KNN (Fixed)", layout="wide")
st.sidebar.title("⚙️ Settings & Controls")

# Upload / UI controls
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=["csv"])
feature_method = st.sidebar.selectbox("🧠 Feature Selection Method", ["Both", "BAT", "CFS"])
k_value = st.sidebar.slider("🔢 K Value for KNN", 1, 15, 7)
test_size = st.sidebar.slider("📊 Test Size (%)", 10, 50, 20, step=5) / 100
run_analysis = st.sidebar.button("🚀 Train Model & Compare")
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", True)
show_metrics_chart = st.sidebar.checkbox("📊 Show Precision/Recall/F1 Chart", True)
show_confusion = st.sidebar.checkbox("📉 Show Confusion Matrices", True)
show_roc_curve = st.sidebar.checkbox("📊 Show ROC Curve", True)
show_distribution_plots = st.sidebar.checkbox("📊 Show Feature Distributions", True)
show_pairplot = st.sidebar.checkbox("🔗 Show Pair Plot", True)
show_diagnostics = st.sidebar.checkbox("🧾 Show Diagnostics", True)

# ---------------- Feature selection helpers ----------------
def cfs_feature_selection(X_df, y, k=6):
    corrs = []
    for i in range(X_df.shape[1]):
        col = X_df.iloc[:, i]
        if col.nunique() < 2:
            corrs.append(0.0)
            continue
        corr = abs(np.corrcoef(col, y)[0,1])
        corrs.append(0.0 if np.isnan(corr) else corr)
    corrs = np.array(corrs)
    topk = np.argsort(corrs)[-min(k, len(corrs)):]
    return topk

def bat_algorithm_feature_selection_cv(X, y, n_starts=12, n_iters=80, cv_folds=5, random_state=42):
    """
    Randomized local-search but evaluate masks with cross_val_score (StratifiedKFold).
    Returns best mask indices.
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    best_mask = None
    best_score = -np.inf

    # Use StratifiedKFold for consistent CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def mask_score(mask):
        sel = np.where(mask == 1)[0]
        if sel.size == 0:
            return -np.inf
        model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        try:
            scores = cross_val_score(model, X[:, sel], y, cv=skf, scoring='accuracy', n_jobs=1)
            return float(np.mean(scores))
        except Exception:
            return -np.inf

    # multiple random starts
    for s in range(n_starts):
        mask = rng.integers(0,2,size=n_features)
        if mask.sum() == 0:
            mask[rng.integers(0,n_features)] = 1
        cur_score = mask_score(mask)
        for it in range(n_iters):
            i = rng.integers(0, n_features)
            new_mask = mask.copy()
            new_mask[i] = 1 - new_mask[i]
            if new_mask.sum() == 0:
                continue
            new_score = mask_score(new_mask)
            if new_score >= cur_score:
                mask = new_mask
                cur_score = new_score
        if cur_score > best_score:
            best_score = cur_score
            best_mask = mask.copy()

    if best_mask is None:
        # fallback to selecting top feature
        return np.array([0])
    return np.where(best_mask == 1)[0]

# ---------------- Training & evaluation ----------------
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    """
    Train model on X_train/y_train and evaluate on X_test/y_test.
    Returns metrics, confusion matrix, trained model and y_pred.
    """
    if len(np.unique(y_train)) < 2:
        majority = int(pd.Series(y_train).mode()[0])
        class DummyModel:
            def predict(self, X): return np.array([majority]*len(X))
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

    acc = round(accuracy_score(y_test, y_pred)*100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0)*100, 2)
    rec = round(recall_score(y_test, y_pred, zero_division=0)*100, 2)
    f1 = round(f1_score(y_test, y_pred, zero_division=0)*100, 2)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm, model, y_pred

# ---------------- Session state init ----------------
if "models" not in st.session_state: st.session_state.models = {}
if "selected_idx" not in st.session_state: st.session_state.selected_idx = {}
if "results" not in st.session_state: st.session_state.results = {}
if "scaler" not in st.session_state: st.session_state.scaler = None
if "X_train_full" not in st.session_state: st.session_state.X_train_full = None

# ---------------- Load dataset ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset")
else:
    df = pd.read_csv("heart.csv")
    st.subheader("📄 Default Dataset Loaded (heart.csv)")

if "target" not in df.columns:
    st.error("❌ Dataset must contain a 'target' column.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

X_df = df.drop("target", axis=1)
y = df["target"].values

# Fit scaler (important: fit on full dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
st.session_state.scaler = scaler

# Create train/test splits (stratified)
X_train_full, X_test_full, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y, random_state=42)
st.session_state.X_train_full = X_train_full
st.session_state.X_test_full = X_test_full
st.session_state.y_train = y_train
st.session_state.y_test = y_test

# ---------------- Run analysis ----------------
if run_analysis:
    st.session_state.models = {}
    st.session_state.selected_idx = {}
    st.session_state.results = {}

    # Diagnostics: class balance
    if show_diagnostics:
        st.markdown("### Diagnostics — class balance")
        st.write("Full dataset target counts:", pd.Series(y).value_counts().to_dict())
        st.write("Train target counts:", pd.Series(y_train).value_counts().to_dict())
        st.write("Test target counts:", pd.Series(y_test).value_counts().to_dict())

    # BAT
    if feature_method in ["BAT", "Both"]:
        st.info("Running BAT feature selection (CV-based search). This may take a bit...")
        bat_idx = bat_algorithm_feature_selection_cv(X_train_full, y_train, n_starts=12, n_iters=80, cv_folds=5, random_state=42)
        if bat_idx.size == 0:
            st.warning("BAT selected no features; falling back to top-6 CFS.")
            bat_idx = cfs_feature_selection(pd.DataFrame(X_train_full, columns=X_df.columns), y_train, k=6)
        # train/eval
        X_tr_b, X_te_b = X_train_full[:, bat_idx], X_test_full[:, bat_idx]
        bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_model, bat_y_pred = train_and_evaluate(X_tr_b, X_te_b, y_train, y_test, k_value)
        # cross-validated score of the chosen subset (for info)
        try:
            cv_model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
            cv_scores = cross_val_score(cv_model, X_train_full[:, bat_idx], y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
            bat_cv_mean = round(float(np.mean(cv_scores))*100,2)
        except Exception:
            bat_cv_mean = None

        st.session_state.results["BAT"] = [bat_acc, bat_prec, bat_rec, bat_f1, bat_cm, bat_idx, bat_cv_mean]
        st.session_state.models["BAT"] = bat_model
        st.session_state.selected_idx["BAT"] = bat_idx

    # CFS
    if feature_method in ["CFS", "Both"]:
        cfs_idx = cfs_feature_selection(X_df, y, k=min(6, X_df.shape[1]))
        X_tr_c, X_te_c = X_train_full[:, cfs_idx], X_test_full[:, cfs_idx]
        cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_model, cfs_y_pred = train_and_evaluate(X_tr_c, X_te_c, y_train, y_test, k_value)
        try:
            cv_model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
            cv_scores = cross_val_score(cv_model, X_train_full[:, cfs_idx], y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
            cfs_cv_mean = round(float(np.mean(cv_scores))*100,2)
        except Exception:
            cfs_cv_mean = None

        st.session_state.results["CFS"] = [cfs_acc, cfs_prec, cfs_rec, cfs_f1, cfs_cm, cfs_idx, cfs_cv_mean]
        st.session_state.models["CFS"] = cfs_model
        st.session_state.selected_idx["CFS"] = cfs_idx

    # Full model
    full_acc, full_prec, full_rec, full_f1, full_cm, full_model, full_y_pred = train_and_evaluate(X_train_full, X_test_full, y_train, y_test, k_value)
    try:
        cv_model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
        cv_scores = cross_val_score(cv_model, X_train_full, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        full_cv_mean = round(float(np.mean(cv_scores))*100,2)
    except Exception:
        full_cv_mean = None

    st.session_state.results["Full"] = [full_acc, full_prec, full_rec, full_f1, full_cm, np.arange(X_train_full.shape[1]), full_cv_mean]
    st.session_state.models["Full"] = full_model
    st.session_state.selected_idx["Full"] = np.arange(X_train_full.shape[1])

    st.success("✅ Analysis complete — models trained and saved to session state.")

# ---------------- Visualizations & writeups ----------------
results = st.session_state.get("results", {})
if results:
    st.subheader("📈 Model Comparison & Visualizations")

    # show selected features and CV info
    for method in results:
        idx = results[method][5]
        cv_mean = results[method][6] if len(results[method])>6 else None
        feat_names = list(X_df.columns[list(idx)])
        st.markdown(f"**{method} selected ({len(feat_names)} features):** {', '.join(feat_names)}")
        if cv_mean is not None:
            st.write(f"Cross-validated (train) mean accuracy for this feature set: **{cv_mean}%**")

    # Accuracy gauge
    if show_accuracy_chart:
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=results[method][0],
                title={"text": f"{method} Accuracy (%)"},
                gauge={"axis": {"range":[0,100]}}
            ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation:** Accuracy on test set shown above.")

    # Metrics chart
    if show_metrics_chart:
        metrics = ["Precision","Recall","F1"]
        fig = go.Figure()
        for method in results:
            fig.add_trace(go.Bar(x=metrics, y=results[method][1:4], name=method))
        fig.update_layout(barmode='group', title="Precision / Recall / F1 (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices + prediction distribution diagnostics
    if show_confusion:
        for method in results:
            st.subheader(f"{method} Confusion Matrix")
            cm = results[method][4]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close(fig)

            # prediction distribution on test set
            model = st.session_state.models.get(method)
            idx = st.session_state.selected_idx.get(method)
            X_test_sub = X_test_full[:, idx]
            preds = model.predict(X_test_sub)
            st.write(f"{method} test predictions distribution:", pd.Series(preds).value_counts().to_dict())
            st.write(f"{method} test true distribution:", pd.Series(y_test).value_counts().to_dict())
            if len(np.unique(preds)) == 1:
                st.warning(f"⚠️ {method} predicts only one class on the test set. Consider lower k, more features, or another classifier.")

    # ROC curves
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
            ax.text(0.5,0.5,"No model supports predict_proba", ha='center')
        ax.plot([0,1],[0,1], linestyle='--', color='red')
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend()
        st.pyplot(fig); plt.close(fig)

# distributions & pairplot (kept)
if show_distribution_plots:
    if 'age' in df.columns:
        st.subheader("Age distribution by target")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df, x='age', hue='target', multiple='stack', ax=ax)
        st.pyplot(fig); plt.close(fig)

if show_pairplot:
    subset_cols = [c for c in ['age','chol','thalach','target'] if c in df.columns]
    if len(subset_cols) >= 2:
        st.subheader("Pair plot")
        pair_fig = sns.pairplot(df[subset_cols], hue='target').fig
        st.pyplot(pair_fig); plt.close(pair_fig)

# ---------------- Real-time prediction ----------------
st.subheader("🔍 Real-Time Prediction")
st.markdown("Pick which trained model to use for prediction (BAT / CFS / Full). If model predicts only one class, choose another model or retrain with different settings.")

with st.form("patient_form"):
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male","Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes","No"])
    restecg = st.selectbox("Resting ECG", ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes","No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping","Flat","Downsloping"])
    ca = st.number_input("Number of Major Vessels", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversible Defect"])
    model_options = list(st.session_state.models.keys()) if st.session_state.models else ["Full"]
    chosen_model = st.selectbox("Model to use for prediction", options=model_options)
    submit_button = st.form_submit_button("📈 Predict Now")

if submit_button:
    maps = {
        "sex": {"Male":1,"Female":0},
        "cp": {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3},
        "fbs": {"Yes":1,"No":0},
        "restecg": {"Normal":0,"ST-T Wave Abnormality":1,"Left Ventricular Hypertrophy":2},
        "exang": {"Yes":1,"No":0},
        "slope": {"Upsloping":0,"Flat":1,"Downsloping":2},
        "thal": {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}
    }

    patient_row = pd.DataFrame([[
        age, maps["sex"][sex], maps["cp"][cp], trestbps, chol, maps["fbs"][fbs],
        maps["restecg"][restecg], thalach, maps["exang"][exang], oldpeak,
        maps["slope"][slope], ca, maps["thal"][thal]
    ]], columns=X_df.columns)

    input_scaled = scaler.transform(patient_row)

    model_key = chosen_model if chosen_model in st.session_state.models else "Full"
    model = st.session_state.models.get(model_key)
    idx = st.session_state.selected_idx.get(model_key, np.arange(X_train_full.shape[1]))

    if model is None:
        st.error("No trained model available. Run Train & Compare first.")
    else:
        inp = input_scaled[:, idx]
        # ensure predict_proba
        if not hasattr(model, "predict_proba"):
            class Wrapped:
                def __init__(self, m): self.m = m
                def predict(self, X): return self.m.predict(X)
                def predict_proba(self, X):
                    preds = self.m.predict(X)
                    probs = np.zeros((len(X),2))
                    for i,p in enumerate(preds): probs[i,p] = 1.0
                    return probs
            model = Wrapped(model)

        pred = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        if pred == 1:
            st.error(f"🛑 Positive (Heart Disease) — Confidence: {max(proba)*100:.2f}%")
        else:
            st.success(f"✅ Negative (No Heart Disease) — Confidence: {max(proba)*100:.2f}%")

        st.markdown(f"**Model used:** {model_key}")
        st.markdown(f"**Features used ({len(idx)}):** {', '.join(list(X_df.columns[idx]))}")

# ---------------- Footer notes ----------------
st.markdown("---")
st.markdown("""
**Notes & next steps**

- If a model predicts **only the majority class** on the test set (you'll see that in the diagnostics), try:
  - Lower **k** (1 or 3).  
  - Use the **Full** model or a different model (RandomForest / LogisticRegression).  
  - Use oversampling (SMOTE) or class weights — I can add this if you want.

- CV mean accuracy for a feature set is reported next to the selected features — that is a better indication of expected performance than a single holdout.

If you run this and still always get Negative:
1. Paste the diagnostics output (train/test class counts and each model's test prediction distribution) here and I’ll pinpoint the exact root cause.
2. Tell me which test accuracy numbers you *expect* (e.g., 85.00, 75.00) and how you computed them previously so I can reproduce that exact evaluation protocol.
""")
