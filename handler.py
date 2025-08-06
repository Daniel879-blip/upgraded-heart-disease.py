import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import (
    bat_algorithm_feature_selection,
    cfs_feature_selection,
    train_and_evaluate
)


def load_data(path="heart.csv"):
    return pd.read_csv(path)


def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, scaler


def split_data(X_scaled, y, test_size=0.2):
    return train_test_split(X_scaled, y, test_size=test_size, stratify=y, random_state=42)


def train_model_with_feature_selection(X_df, X_scaled, y, method="BAT", k_value=5, test_size=0.2):
    # Split the full scaled dataset
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size)

    selected_idx = None

    if method == "BAT":
        selected_idx = bat_algorithm_feature_selection(X_train, y_train)
    elif method == "CFS":
        selected_idx = cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train)
    elif method == "Both":
        # Combine top BAT and top CFS (union of both)
        bat_idx = set(bat_algorithm_feature_selection(X_train, y_train))
        cfs_idx = set(cfs_feature_selection(pd.DataFrame(X_train, columns=X_df.columns), y_train))
        selected_idx = list(bat_idx.union(cfs_idx))
    else:
        selected_idx = list(range(X_train.shape[1]))  # fallback: all features

    # Select features based on indices
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # Train and evaluate
    metrics = train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test, k_value)

    return {
        "model": metrics["model"],
        "selected_idx": selected_idx,
        "scaler": StandardScaler().fit(X_df),  # For consistent real-time scaling
        "metrics": metrics,
        "X_train": X_train_sel,
        "X_test": X_test_sel,
        "y_train": y_train,
        "y_test": y_test
    }
