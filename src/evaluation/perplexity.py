import math

def perplexity(model, tokens, n=4):
    """
    Compute perplexity over tokens using rolling history.
    """
    log_sum = 0.0
    count = 0

    for i in range(n-1, len(tokens)):
        history = tokens[i-(n-1):i]
        word = tokens[i]
        p = model.prob(word, history)
        log_sum += math.log(p)
        count += 1

    return math.exp(-log_sum / max(count, 1))
