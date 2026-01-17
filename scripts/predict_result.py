import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Imports
from scripts.feature_extraction import extract_features
import joblib

# Load trained model
model = joblib.load(ROOT_DIR / "lr_model.pkl")

# User input
review = input("Enter a review: ")

# Predict sentiment
features = [extract_features(review)]
prediction = model.predict(features)

print("\n Prediction Result")
print("-------------------")
if prediction[0] == 1:
    print("Sentiment: POSITIVE")
else:
    print("Sentiment: NEGATIVE")
