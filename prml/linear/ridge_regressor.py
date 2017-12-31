# https://github.com/ctgk/PRML/blob/master/prml/linear/ridge_regressor.py

import numpy as np
from prml.linear.regressor import Regressor


class RidgeRegressor(Regressor):
    """
    Ridge regression model
    w* = argmin |t - X @ w| + a * |w|_2^2
    """

    def __init__(self, alpha=1.):
        self.alpha = alpha
    
    def _fit(self, X, t):
        eye = np.eye(np.size(X, axis=1))
        '''
        np.linalg.solve(A, b)
        Solve a linear matrix equation: Ax = b
        '''
        # (3.28) w = (\lambda * I + \Phi^\top * \Phi)^{-1} * \Phi^\top * t
        # w is obtained by solving the equation below
        # (3.28') (\lambda * I + \Phi^\top * \Phi) * w = \Phi^\top * t
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)
    
    def _predict(self, X):
        # (3.3') y(X, w) = \Phi * w
        y = X @ self.w
        return y
