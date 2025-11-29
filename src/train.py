# src/train.py
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

from .models import build_models
from .config import MODEL_DIR, REPORT_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def _save_confusion_matrix(y_true, y_pred, label, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {label}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def _save_roc_curve(pipe, X_test, y_test, label, filename):
    if hasattr(pipe.named_steps["classifier"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"ROC Curve - {label}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test):
    pipelines = build_models(preprocessor)
    results = {}

    for name, pipe in pipelines.items():
        print(f"🤖 Training: {name}")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps["classifier"], "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        print(f"📊 {name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, ROC_AUC: {roc if not np.isnan(roc) else 'N/A'}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipe, model_path)
        print(f"💾 Saved {name} model to {model_path}")

        # Save confusion matrix and ROC
        cm_path = os.path.join(REPORT_DIR, f"confusion_{name}.png")
        _save_confusion_matrix(y_test, y_pred, name, cm_path)
        print(f"💾 Saved confusion matrix to {cm_path}")

        roc_path = os.path.join(REPORT_DIR, f"roc_{name}.png")
        _save_roc_curve(pipe, X_test, y_test, name, roc_path)
        if os.path.exists(roc_path):
            print(f"💾 Saved ROC curve to {roc_path}")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc
        }

    return results
