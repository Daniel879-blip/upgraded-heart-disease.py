# handler.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("heart.csv")

def preprocess_data(df, test_size=0.2):
    X_df = df.drop("target", axis=1)
    y = df["target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    return X_df, scaler, X_train, X_test, y_train, y_test
