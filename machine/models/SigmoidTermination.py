import numpy as np
from scipy.special import expit


class SigmoidTermination():
    def __init__(self, rng, lr, nstates):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.weights = np.zeros((nstates,))

    def pmf(self, state):
        return expit(self.weights[state])

    def sample(self, state):
        return int(self.rng.uniform() < self.pmf(state))

    def gradient(self, state):
        return self.pmf(state) * (1.0 - self.pmf(state)), state

    def update(self, state, advantage):
        magnitude, direction = self.gradient(state)
        self.weights[direction] -= self.lr * magnitude * advantage
