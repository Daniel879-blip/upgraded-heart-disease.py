# model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)

# ---------- BAT Algorithm for Feature Selection ---------- #
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

# ---------- CFS Feature Selection ---------- #
def cfs_feature_selection(X_df, y, k=6):
    correlations = [abs(np.corrcoef(X_df.iloc[:, i], y)[0, 1]) for i in range(X_df.shape[1])]
    return np.argsort(correlations)[-k:]

# ---------- Train and Evaluate KNN ---------- #
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

# ---------- ROC Curve Calculation ---------- #
def get_roc_curve(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    return fpr, tpr, auc(fpr, tpr)

# ---------- Real-Time Prediction ---------- #
def predict_new(model, scaler, selected_idx, raw_input):
    input_array = np.array(raw_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_selected = input_scaled[:, selected_idx]
    prediction = model.predict(input_selected)
    return prediction[0]
