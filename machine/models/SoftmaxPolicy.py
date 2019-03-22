import numpy as np
from scipy.misc import logsumexp


class SoftmaxPolicy():
    def __init__(self, rng, lr, nstates, nactions, temperature=1.0):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.nactions = nactions
        self.temperature = temperature
        self.weights = np.zeros((nstates, nactions))

    def Q_U(self, state, action=None):
        if action is None:
            return self.weights[state, :]
        else:
            return self.weights[state, action]

    def pmf(self, state):
        exponent = self.Q_U(state) / self.temperature
        return np.exp(exponent - logsumexp(exponent))

    def sample(self, state):
        return int(self.rng.choice(self.nactions, p=self.pmf(state)))

    def gradient(self):
        pass

    def update(self, state, action, Q_U):
        actions_pmf = self.pmf(state)
        self.weights[state, :] -= self.lr * actions_pmf * Q_U
        self.weights[state, action] += self.lr * Q_U