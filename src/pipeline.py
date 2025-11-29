# src/pipeline.py
from .data_loader import load_data
from .preprocessing import preprocess
from .train import train_and_evaluate
from .config import PROCESSED_DIR
import os

def main():
    print("📥 Loading dataset...")
    df = load_data()

    print("🔧 Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    # Save processed data as CSVs
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    print(f"💾 Preprocessed data saved to {PROCESSED_DIR}")

    print("🤖 Training & evaluating models...")
    results = train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test)

    # Save results summary
    import json
    with open(os.path.join(PROCESSED_DIR, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Results summary saved.")

if __name__ == "__main__":
    main()
