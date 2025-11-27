from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_models(preprocessor):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    return pipelines
