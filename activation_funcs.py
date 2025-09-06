import numpy as np

class ActivationStrategy:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dA):
        raise NotImplementedError


class ReLU(ActivationStrategy):
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        ...

class Softmax(ActivationStrategy):
    def forward(self, X):
        self.X = X
        return np.exp(X) / np.sum(np.exp(X), axis=0)

    def backward(self, dA):
        ...