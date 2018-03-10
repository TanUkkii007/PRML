import numpy as np
from prml.linear.classifier import Classifier
from typing import Union


class LogisticRegressor(Classifier):
    """
    Logistic regression model
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def _fit(self, X: np.ndarray, t: np.ndarray, max_iter=100):
        self._check_binary(t)
        w = np.zeros(np.size(X, axis=1))
        # IRLS (iterative reweighted least squares) based on
        # Newton-Raphson method
        for _ in range(max_iter):
            w_prev = np.copy(w)
            # (4.87) p(C_1|\phi) = y(\phi) = \sigma(w^\top \phi)
            y = self._sigmoid(X @ w)
            # (4.96) \nabla E(w) = \Phi^\top (y - t)
            grad = X.T @ (y - t)
            # (4.97) H = \Phi^\top R \Phi
            # (4.98) R_nn = y_n (1 - y_n)
            hessian = (X.T * y * (1 - y) @ X)
            try:
                # (4.92) w' = w - H^{-1} \nabla E(w)
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w


    def _sigmoid(self, a: Union[np.ndarray, float]):
        # \sigma(a) = 1/2 \tanh(a/2) + 1/2
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _proba(self, X: np.ndarray):
        y = self._sigmoid(X @ self.w)
        return y

    def _classify(self, X: np.ndarray, threshold=0.5):
        proba = self._proba(X)
        label = (proba > threshold).astype(np.int)
        return label