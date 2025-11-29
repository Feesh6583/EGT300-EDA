# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def build_models(preprocessor):
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "grad_boost": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
    return pipelines
