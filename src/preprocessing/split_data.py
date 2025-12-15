import random

with open("data/raw/corpus.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


random.shuffle(lines)

# Split
total = len(lines)
train_end = int(total * 0.7)
valid_end = int(total * 0.8)  #70% + 10% = 80%

train_lines = lines[:train_end]
valid_lines = lines[train_end:valid_end]
test_lines = lines[valid_end:]

# Save splits
with open("data/processed/train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("data/processed/valid.txt", "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

with open("data/processed/test.txt", "w", encoding="utf-8") as f:
    f.writelines(test_lines)

print("Corpus split completed!")
