from collections import Counter

def load_tokens(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split()

def make_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

class NgramCounter:
    """
    Stores counts for 1-gram..4-gram and context counts.
    """
    def __init__(self, tokens, max_n=4):
        self.max_n = max_n
        self.counts = {n: Counter(make_ngrams(tokens, n)) for n in range(1, max_n+1)}

        # context_counts[n][context] = total count of that context (length n-1)
        self.context_counts = {}
        for n in range(2, max_n+1):
            ctx = Counter()
            for ng, c in self.counts[n].items():
                ctx[ng[:-1]] += c
            self.context_counts[n] = ctx

        self.total_unigrams = sum(self.counts[1].values())
        self.vocab = set(w for (w,) in self.counts[1].keys())
        self.V = len(self.vocab)
