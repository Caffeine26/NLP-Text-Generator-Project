import re
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.load_data import load_reviews


# Load sentiment word lists
positive_words = set(load_reviews("positive-words.txt"))
negative_words = set(load_reviews("negative-words.txt"))

def extract_features(review):
    """
    Extract features from a single review
    """
    words = re.findall(r"\b\w+\b", review.lower())

    # REQUIRED FEATURES
    positive_count = sum(1 for w in words if w in positive_words)
    negative_count = sum(1 for w in words if w in negative_words)

    contains_no = 1 if "no" in words else 0

    pronouns = ["i", "me", "my", "you", "your"]
    pronoun_count = sum(1 for w in words if w in pronouns)

    contains_exclamation = 1 if "!" in review else 0
    log_review_length = math.log(len(words) + 1)

    # EXTRA FEATURES (to improve performance)
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / (len(words) + 1)
    uppercase_words = sum(1 for w in review.split() if w.isupper())
    contains_question = 1 if "?" in review else 0
    sentiment_ratio = positive_count / (negative_count + 1)

    return [
        positive_count,
        negative_count,
        contains_no,
        pronoun_count,
        contains_exclamation,
        log_review_length,
        word_count,
        avg_word_length,
        uppercase_words,
        contains_question,
        sentiment_ratio
    ]
