# model.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# BAT Feature Selection
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
            model = KNeighborsClassifier()
            model.fit(X[:, selected], y)
            fitness[i] = accuracy_score(y, model.predict(X[:, selected]))

    best_bat = population[np.argmax(fitness)]
    return np.where(best_bat == 1)[0]

# CFS Feature Selection
def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    return np.argsort(correlations)[-k:]

# Train and Evaluate
def train_and_evaluate(X_train, X_test, y_train, y_test, k_value=7):
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred
    }

# Real-time Prediction
def predict_new(model, scaler, selected_idx, raw_input, feature_names):
    import pandas as pd

    input_df = pd.DataFrame([raw_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    if selected_idx is not None:
        prediction = model.predict(input_scaled[:, selected_idx])[0]
        proba = model.predict_proba(input_scaled[:, selected_idx])[0]
    else:
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

    return prediction, max(proba) * 100
