from .data_loader import load_data
from .preprocessing import preprocess
from .train import train_and_evaluate
import os
import pandas as pd
from .config import DB_PATH

def main():
    print("📥 Loading dataset...")
    df = load_data()

    print("🔧 Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    # Save preprocessed datasets
    os.makedirs(os.path.join(os.path.dirname(DB_PATH), "processed"), exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("💾 Preprocessed datasets saved in data/processed/")

    print("🤖 Training models...")
    train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test)

    print("\n✅ Pipeline complete.")

if __name__ == "__main__":
    main()
