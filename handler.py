# handler.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("heart.csv")  # fallback local file

def map_input_features(inputs):
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    return pd.DataFrame([[
        inputs['age'],
        sex_map[inputs['sex']],
        cp_map[inputs['cp']],
        inputs['trestbps'],
        inputs['chol'],
        fbs_map[inputs['fbs']],
        restecg_map[inputs['restecg']],
        inputs['thalach'],
        exang_map[inputs['exang']],
        inputs['oldpeak'],
        slope_map[inputs['slope']],
        inputs['ca'],
        thal_map[inputs['thal']]
    ]], columns=inputs['columns'])

def real_time_predict(model, input_scaled, selected_idx=None):
    if selected_idx is not None:
        prediction = model.predict(input_scaled[:, selected_idx])[0]
        proba = model.predict_proba(input_scaled[:, selected_idx])[0]
    else:
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
    return prediction, proba
