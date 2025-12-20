EPS = 1e-12  # tiny fallback so perplexity doesn't become infinity

class BackoffLM:
    """
    LM1: Backoff model (unsmoothed).
    Try 4-gram -> 3-gram -> 2-gram -> 1-gram.
    """
    def __init__(self, ncounter):
        self.nc = ncounter

    def prob(self, word, history):
        # history: list of previous words
        h = tuple(history[-3:])  # need up to 3 words for 4-gram

        # 4-gram
        if len(h) == 3:
            ng4 = h + (word,)
            c4 = self.nc.counts[4].get(ng4, 0)
            if c4 > 0:
                return c4 / self.nc.context_counts[4][h]

        # 3-gram
        if len(h) >= 2:
            h3 = h[-2:]
            ng3 = h3 + (word,)
            c3 = self.nc.counts[3].get(ng3, 0)
            if c3 > 0:
                return c3 / self.nc.context_counts[3][h3]

        # 2-gram
        if len(h) >= 1:
            h2 = (h[-1],)
            ng2 = h2 + (word,)
            c2 = self.nc.counts[2].get(ng2, 0)
            if c2 > 0:
                return c2 / self.nc.context_counts[2][h2]

        # 1-gram
        c1 = self.nc.counts[1].get((word,), 0)
        if c1 > 0:
            return c1 / self.nc.total_unigrams

        return EPS
