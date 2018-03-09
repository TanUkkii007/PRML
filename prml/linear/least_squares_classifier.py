import numpy as np
from prml.linear.classifier import Classifier


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier model
    y = argmax_k X @ W
    """

    def __init__(self, W=None):
        self.W = W

    def _fit(self, X, t):
        self._check_input(X)
        self._check_target(t)
        T = np.eye(int(np.max(t)) + 1)[t]
        # (4.17) y = W^\top x = T^\top (X^+)^\top x
        # (4.17') W = X^+ T
        self.W = np.linalg.pinv(X) @ T

    def _classify(self, X):
        # (4.14) y = W^\top x
        return np.argmax(X @ self.W, axis=-1)