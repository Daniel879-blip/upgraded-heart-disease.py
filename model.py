# model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler


def bat_algorithm_feature_selection(X, y, n_bats=8, n_iterations=8):
    n_features = X.shape[1]
    rng = np.random.default_rng(42)
    population = rng.integers(0, 2, size=(n_bats, n_features))
    fitness = np.zeros(n_bats)

    for i in range(n_bats):
        selected = np.where(population[i] == 1)[0]
        if len(selected) == 0:
            fitness[i] = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, selected], y, test_size=0.2, stratify=y, random_state=42
            )
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            fitness[i] = accuracy_score(y_test, model.predict(X_test))

    best_bat = population[np.argmax(fitness)].copy()
    return np.where(best_bat == 1)[0]


def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    return np.argsort(correlations)[-k:]


def train_and_evaluate(X_train, X_test, y_train, y_test, k_value):
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return (
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        confusion_matrix(y_test, y_pred),
        model,
        y_pred
    )


def predict_real_time(input_scaled, model, selected_idx):
    """
    Predict for real-time patient input.
    """
    if selected_idx is not None and len(selected_idx) > 0:
        input_selected = input_scaled[:, selected_idx]
    else:
        input_selected = input_scaled

    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0]
    return prediction, probability
