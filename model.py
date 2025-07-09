
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }
    results = []
    for name, model in models.items():
        
        model.fit(X, y)
        y_pred = model.predict(X)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1-Score": f1_score(y, y_pred)
        })
    return results

def predict_new(data):
    import pandas as pd
    clf = RandomForestClassifier()
    import pandas as pd
    df = pd.read_csv("heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    clf.fit(X, y)
    return clf.predict(data)[0]
