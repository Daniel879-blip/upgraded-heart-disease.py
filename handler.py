import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load CSV or fallback to default
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("heart.csv")
    return df

# Scale and split data
def preprocess_data(df, test_size=0.2):
    X_df = df.drop("target", axis=1)
    y = df["target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )
    return X_df, scaler, X_train, X_test, y_train, y_test

# Map and transform patient input for prediction
def transform_patient_input(inputs, X_df_columns):
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Yes": 1, "No": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    data = [[
        inputs["age"], sex_map[inputs["sex"]], cp_map[inputs["cp"]],
        inputs["trestbps"], inputs["chol"], fbs_map[inputs["fbs"]],
        restecg_map[inputs["restecg"]], inputs["thalach"], exang_map[inputs["exang"]],
        inputs["oldpeak"], slope_map[inputs["slope"]], inputs["ca"], thal_map[inputs["thal"]]
    ]]
    return pd.DataFrame(data, columns=X_df_columns)

# Predict based on selected features (BAT or fallback)
def predict_patient(model, input_df, scaler, selected_features=None):
    scaled_input = scaler.transform(input_df)
    if selected_features is not None:
        scaled_input = scaled_input[:, selected_features]
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    return prediction, max(proba) * 100
