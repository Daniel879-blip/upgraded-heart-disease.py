import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Load Dataset ---------- #
def load_data(uploaded_file=None):
    """
    Loads dataset from an uploaded file (if provided), otherwise loads 'heart.csv'.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # file-like object
    else:
        df = pd.read_csv("heart.csv")  # fallback local file
    return df

# ---------- Preprocess Dataset ---------- #
def preprocess_data(df, test_size=0.2):
    """
    Splits dataset into scaled train and test sets.
    Returns: original features (X_df), scaler, and train/test splits.
    """
    X_df = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )
    return X_df, scaler, X_train, X_test, y_train, y_test
