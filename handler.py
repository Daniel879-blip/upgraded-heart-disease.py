import pandas as pd

def load_data():
    return pd.read_csv("heart.csv")

def preprocess_data(df):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler
    X = df.drop("target", axis=1)
    y = df["target"]
    selector = SelectKBest(f_classif, k=8)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X[selected_features]), columns=selected_features)
    return X_scaled, y, selected_features
