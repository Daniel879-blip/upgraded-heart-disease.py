import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

def load_data():
    df = pd.read_csv("heart.csv")
    return df

def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # 🔹 Show accuracy as percentage
    st.write(f"**Accuracy:** {acc * 100:.2f}%")

    # Optional detailed output
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    return model
