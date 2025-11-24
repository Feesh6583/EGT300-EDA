# src/models.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def build_models(preprocessor):
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    return pipelines

def train_models(pipelines, X_train, y_train):
    trained_pipelines = {}
    for name, pipe in pipelines.items():
        print(f"🔵 Training: {name}")
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
    return trained_pipelines

def evaluate_models(trained_pipelines, X_test, y_test):
    for name, pipe in trained_pipelines.items():
        y_pred = pipe.predict(X_test)
        print(f"\n📊 {name} Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
