import random

def _sample(model, history, vocab):
    probs = [model.prob(w, history) for w in vocab]
    s = sum(probs)
    if s <= 0:
        return random.choice(vocab)

    r = random.random()
    cum = 0.0
    for w, p in zip(vocab, probs):
        cum += p / s
        if r <= cum:
            return w
    return vocab[-1]

def generate(model, seed, vocab, max_words=50):
    out = list(seed)
    while len(out) < max_words:
        history = out[-3:]
        nxt = _sample(model, history, vocab)
        out.append(nxt)
        if nxt == "</s>":  # if you used sentence end token
            break
    return " ".join(out)
