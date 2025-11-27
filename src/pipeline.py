from .data_loader import load_data
from .preprocess import preprocess_data
from .models import get_models
from .config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("📥 Loading dataset...")
    df = load_data()

    print("🔧 Preprocessing data...")
    df = preprocess_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = get_models()

    print("🤖 Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"📊 {name} Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
