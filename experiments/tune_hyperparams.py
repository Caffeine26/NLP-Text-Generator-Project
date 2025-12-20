from src.models.ngram_base import load_tokens, NgramCounter
from src.models.backoff_lm import BackoffLM
from src.models.interpolated_lm import InterpolatedAddKLM
from src.evaluation.perplexity import perplexity
from src.generation.text_generator import generate
import os

TRAIN = "data/processed/train_final.txt"
VALID = "data/processed/valid_final.txt"
TEST  = "data/processed/test_final.txt"

os.makedirs("results", exist_ok=True)

train_tokens = load_tokens(TRAIN)
valid_tokens = load_tokens(VALID)
test_tokens  = load_tokens(TEST)

nc = NgramCounter(train_tokens, max_n=4)

lm1 = BackoffLM(nc)
lm2 = InterpolatedAddKLM(nc, lambdas=(0.4,0.3,0.2,0.1), k=0.1)

pp_lm1 = perplexity(lm1, test_tokens)
pp_lm2 = perplexity(lm2, test_tokens)

# Save perplexity
with open("results/perplexity_scores.txt", "w", encoding="utf-8") as f:
    f.write(f"LM1 (Backoff, unsmoothed) Perplexity: {pp_lm1}\n")
    f.write(f"LM2 (Interpolated + add-k) Perplexity: {pp_lm2}\n")

# Generate text (2-50 words)
vocab = list(nc.vocab)
seed = ["the", "border", "dispute"]  # you can change this seed

gen1 = generate(lm1, seed, vocab, max_words=50)
gen2 = generate(lm2, seed, vocab, max_words=50)

with open("results/generated_text.txt", "w", encoding="utf-8") as f:
    f.write("LM1 Generated:\n" + gen1 + "\n\n")
    f.write("LM2 Generated:\n" + gen2 + "\n")

print("DONE ✅ Check results/perplexity_scores.txt and results/generated_text.txt")
