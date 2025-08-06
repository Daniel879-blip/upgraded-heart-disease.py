import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from model import (
    bat_algorithm_feature_selection,
    cfs_feature_selection,
    train_and_evaluate,
    predict_new
)

def load_data(path="heart.csv"):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, scaler

def train_model_with_feature_selection(X_df, X_scaled, y, method="BAT", k_value=7, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    if method == "BAT":
        selected_idx = bat_algorithm_feature_selection(X_train, y_train)
    elif method == "CFS":
        selected_idx = cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train)
    else:
        selected_idx = list(range(X_train.shape[1]))

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    result = train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test, k_value)

    return {
        "model": result["model"],
        "selected_idx": selected_idx,
        "metrics": {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "conf_matrix": result["conf_matrix"]
        }
    }

def predict_patient(model, scaler, selected_idx, patient_df):
    return predict_new(model, scaler, selected_idx, patient_df)
