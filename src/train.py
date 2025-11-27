import os
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .models import build_models
from .config import MODEL_DIR

def train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test):
    pipelines = build_models(preprocessor)
    results = {}

    for name, pipe in pipelines.items():
        print(f"🤖 Training: {name}")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["classifier"], "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

        print(f"📊 {name} Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc}")

        results[name] = {"pipeline": pipe, "accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipe, model_path)
        print(f"💾 Saved {name} model to {model_path}")

    return results
