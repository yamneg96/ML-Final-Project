import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

MODEL_DIR = "models/"

def train_logistic(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, name="best_model.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, f"{MODEL_DIR}{name}")
    print(f"Model saved as {MODEL_DIR}{name}")

