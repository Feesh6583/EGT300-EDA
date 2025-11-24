# src/pipeline.py
from .data_loader import load_data
from .preprocess import preprocess_data
from .models import build_models, train_models, evaluate_models

def main():
    print("📥 Loading data...")
    df = load_data()

    print("🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("⚙️ Building models...")
    pipelines = build_models(preprocessor)

    print("🚀 Training models...")
    trained_pipelines = train_models(pipelines, X_train, y_train)

    print("📊 Evaluating models...")
    evaluate_models(trained_pipelines, X_test, y_test)

    print("\n✅ Pipeline complete.")

if __name__ == "__main__":
    main()
