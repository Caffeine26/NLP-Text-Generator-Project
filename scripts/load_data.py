from pathlib import Path

# Path to data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_reviews(filename):
    """
    Load reviews from a text file (one review per line)
    """
    with open(DATA_DIR / filename, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f.readlines()]

# Load datasets
positive_reviews = load_reviews("positive-reviews.txt")
negative_reviews = load_reviews("negative-reviews.txt")

# Check size
print("Positive reviews:", len(positive_reviews))
print("Negative reviews:", len(negative_reviews))

# 80% training, 20% testing
pos_train = positive_reviews[:16000]
pos_test  = positive_reviews[16000:]

neg_train = negative_reviews[:16000]
neg_test  = negative_reviews[16000:]
