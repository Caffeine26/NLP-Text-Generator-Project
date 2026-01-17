import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Imports
from scripts.load_data import pos_test, neg_test
from scripts.feature_extraction import extract_features

from sklearn.metrics import accuracy_score
import joblib

# Load trained models
lr_model = joblib.load(ROOT_DIR / "lr_model.pkl")
nb_model = joblib.load(ROOT_DIR / "nb_model.pkl")

# Build test data
X_test = [extract_features(r) for r in pos_test + neg_test]
y_test = [1] * len(pos_test) + [0] * len(neg_test)

# Evaluate models
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

print("Evaluation Results")
print("---------------------")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Naive Bayes Accuracy:        {nb_accuracy:.4f}")
