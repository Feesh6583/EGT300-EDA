# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

TARGET_COL = "Subscription Status"

def preprocess_data(df):
    df = df.copy()

    # Convert target to numeric
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})

    # Identify features
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != TARGET_COL]

    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # Fill missing values
    df[numeric_features] = df[numeric_features].fillna(0)
    df[categorical_features] = df[categorical_features].fillna("Unknown")

    # Split features/target
    X = df[numeric_features + categorical_features]
    y = df[TARGET_COL]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
