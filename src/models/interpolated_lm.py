class InterpolatedAddKLM:
    """
    LM2: Interpolation + add-k smoothing.
    P = l4*P4 + l3*P3 + l2*P2 + l1*P1
    """
    def __init__(self, ncounter, lambdas=(0.4, 0.3, 0.2, 0.1), k=0.1):
        self.nc = ncounter
        self.l4, self.l3, self.l2, self.l1 = lambdas
        self.k = k
        self.V = ncounter.V

    def _addk(self, word, history, n):
        if n == 1:
            c = self.nc.counts[1].get((word,), 0)
            return (c + self.k) / (self.nc.total_unigrams + self.k * self.V)

        ctx = tuple(history[-(n-1):])
        ng = ctx + (word,)
        c_ng = self.nc.counts[n].get(ng, 0)
        c_ctx = self.nc.context_counts[n].get(ctx, 0)
        return (c_ng + self.k) / (c_ctx + self.k * self.V)

    def prob(self, word, history):
        p4 = self._addk(word, history, 4)
        p3 = self._addk(word, history, 3)
        p2 = self._addk(word, history, 2)
        p1 = self._addk(word, history, 1)
        return self.l4*p4 + self.l3*p3 + self.l2*p2 + self.l1*p1
