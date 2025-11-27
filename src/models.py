from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
    }
