import os
from collections import Counter

# Set vocabulary size
VOCAB_SIZE = 10000
UNK_TOKEN = "<UNK>"

def build_vocab(input_file, vocab_size):
    with open(input_file, "r", encoding="utf-8") as f:
        words = f.read().split()

    counter = Counter(words)
    most_common = counter.most_common(vocab_size)
    vocab = set([word for word, freq in most_common])
    return vocab

def replace_unk(input_file, output_file, vocab):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        words = line.strip().split()
        words = [word if word in vocab else UNK_TOKEN for word in words]
        new_lines.append(" ".join(words) + "\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def save_vocab(vocab, vocab_file):
    with open(vocab_file, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")

# Build vocabulary from training data
vocab = build_vocab("data/processed/train_tokenized.txt", VOCAB_SIZE)
save_vocab(vocab, "data/processed/vocabulary.txt")

# Replace rare words with <UNK> in all splits
replace_unk("data/processed/train_tokenized.txt", "data/processed/train_final.txt", vocab)
replace_unk("data/processed/valid_tokenized.txt", "data/processed/valid_final.txt", vocab)
replace_unk("data/processed/test_tokenized.txt", "data/processed/test_final.txt", vocab)

print("Vocabulary created and rare words replaced with <UNK>.")
