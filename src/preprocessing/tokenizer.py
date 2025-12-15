import nltk

# Ensure the punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
import os

def tokenize_file(input_file, output_file):
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tokenized_lines = []
    for line in lines:
        tokens = word_tokenize(line.lower())  # lowercase + tokenize
        tokenized_lines.append(" ".join(tokens) + "\n")

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(tokenized_lines)

    print(f"Tokenization completed: {output_file}")

# Run tokenization for train, validation, and test
tokenize_file("data/processed/train.txt", "data/processed/train_tokenized.txt")
tokenize_file("data/processed/valid.txt", "data/processed/valid_tokenized.txt")
tokenize_file("data/processed/test.txt", "data/processed/test_tokenized.txt")
