# https://github.com/ctgk/PRML/blob/master/prml/linear/bayesian_regressor.py

import numpy as np
from prml.linear.regressor import Regressor


class BayesianRegressor(Regressor):
    """
    Bayesian regression model
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None
    
    def _fit(self, X, t):
        # prior is 
        # (3.52) p(w|\alpha) = N(w|0, \alpha^{-1})
        if self.w_mean is not None:
            mean_prev = self.w_mean
        else:
            mean_prev = np.zeros(np.size(X, 1))
        if self.w_precision is not None:
            precision_prev = self.w_precision
        else:
            precision_prev = self.alpha * np.eye(np.size(X, 1))
        
        # w ~ N(w|m_N, S_N)
        # aftter observing N inputs
        # (3.51) S_N^{-1} = S_0^{-1} + \beta * \Phi^\top * \Phi
        # aftter observing another input
        # (3.51') S_{N+1}^{-1} = S_N^{-1} + \beta \phi(x_{N+1})\phi(x_{N+1})^\top
        # See problem 3.8
        w_precision = precision_prev + self.beta * X.T @ X
        # aftter observing N inputs
        # (3.50) m_N = S_N(S_0^{-1} * m_0 + \beta * \Phi^\top * t) 
        # aftter observing another input
        # (3.50') m_{N+1} = S_{N+1}(m_N^\top * S_N^{-1} + \beta * t_{N+1} * \phi(x_{N+1})^\top) 
        # m_N is obtained by solving the equation below
        # S_N^{-1} * m_N = S_0^{-1} * m_0 + \beta * \Phi^\top * t
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def _predict(self, X, return_std=False, sample_size=None):
        # predictive distribution
        # (3.58) p(t|x, t, \alpha, \beta) = N(t| m_N^\top * \phi(x), \sigma_N(x)^2)
        y = X @ self.w_mean
        if return_std:
            # (3.59) \sigma_N(x)^2 = 1/\beta + \phi(x)^\top * S_N * \phi(x)
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y