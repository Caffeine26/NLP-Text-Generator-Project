import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Imports (NOW STABLE)
from scripts.load_data import pos_train, neg_train
from scripts.feature_extraction import extract_features

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib

# Build training data
X_train = [extract_features(r) for r in pos_train + neg_train]
y_train = [1] * len(pos_train) + [0] * len(neg_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save models to project root
joblib.dump(lr_model, ROOT_DIR / "lr_model.pkl")
joblib.dump(nb_model, ROOT_DIR / "nb_model.pkl")

print("Models trained and saved successfully.")
