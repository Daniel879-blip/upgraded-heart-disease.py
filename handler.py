# handler.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("heart.csv")
    return df

def preprocess_data(df, test_size=0.2):
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    X_df = df.drop("target", axis=1)
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )
    return X_df, X_scaled, X_train, X_test, y_train, y_test, scaler
