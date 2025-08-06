import pandas as pd
import numpy as np  # ✅ Required for array conversion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ========== Load Data ==========
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("heart.csv")
    return df

# ========== Preprocessing ==========
def preprocess_data(df, test_size=0.2):
    X_df = df.drop("target", axis=1)
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_df, scaler, X_train, X_test, y_train, y_test

# ========== Transform Patient Input ==========
def transform_patient_input(input_dict, feature_columns, scaler):
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    data = [
        input_dict["age"],
        sex_map[input_dict["sex"]],
        cp_map[input_dict["cp"]],
        input_dict["trestbps"],
        input_dict["chol"],
        fbs_map[input_dict["fbs"]],
        restecg_map[input_dict["restecg"]],
        input_dict["thalach"],
        exang_map[input_dict["exang"]],
        input_dict["oldpeak"],
        slope_map[input_dict["slope"]],
        input_dict["ca"],
        thal_map[input_dict["thal"]]
    ]

    df = pd.DataFrame([data], columns=feature_columns)
    scaled = scaler.transform(df)
    return np.array(scaled)  # ✅ Ensure it's a NumPy array

# ========== Predict Patient ==========
def predict_patient(model, scaled_input, selected_idx=None):
    if selected_idx is not None:
        scaled_input = scaled_input[:, selected_idx]  # ✅ Now this won't error
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0]
    return prediction, confidence
