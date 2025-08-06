from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Report": classification_report(y, y_pred, output_dict=True),
        "Confusion Matrix": confusion_matrix(y, y_pred)
    }
    return model, metrics

def evaluate_model(model, X, y, metrics):
    import streamlit as st
    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", metrics["Accuracy"])
    st.write("Classification Report:", metrics["Report"])
    st.write("Confusion Matrix:", metrics["Confusion Matrix"])

def predict_single(model, user_input):
    import pandas as pd
    input_df = pd.DataFrame([user_input])
    return model.predict(input_df)[0]

def predict_batch(model, df):
    return model.predict(df)
