# handler.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import streamlit as st
from model import bat_algorithm_feature_selection, cfs_feature_selection, train_and_evaluate


def load_data(path="heart.csv"):
    return pd.read_csv(path)


def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, scaler


def train_model_with_feature_selection(X_df, X_scaled, y, method="BAT", k_value=5, test_size=0.2):
    # Split full scaled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    if method == "BAT":
        selected_idx = bat_algorithm_feature_selection(X_train, y_train)
    elif method == "CFS":
        selected_idx = cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train)
    else:
        selected_idx = list(range(X_train.shape[1]))  # use all features

    # Select features
    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]

    # Train model
    acc, prec, rec, f1, cm, model, y_pred = train_and_evaluate(
        X_train_selected, X_test_selected, y_train, y_test, k_value
    )

    # Display metrics
    st.write(f"**Accuracy:** {acc:.2f}%")
    st.write(f"**Precision:** {prec:.2f}%")
    st.write(f"**Recall:** {rec:.2f}%")
    st.write(f"**F1 Score:** {f1:.2f}%")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    return model, selected_idx
