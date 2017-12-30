# https://github.com/ctgk/PRML/blob/master/prml/linear/linear_regressor.py

import numpy as np
from prml.linear.regressor import Regressor


class LinearRegressor(Regressor):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def _fit(self, X, t):
        '''
        np.linalg.pinv(array_like)
        Compute the (Moore-Penrose) pseudo-inverse of a matrix.

        Calculate the generalized inverse of a matrix using its
        singular-value decomposition (SVD) and including all
        *large* singular values.
        '''
        # (3.15) w_{ML} = \Phi^+ * t
        # (3.17) \Phi^+ = (\Phi^\top\Phi)^{-1}\Phi^\top
        self.w = np.linalg.pinv(X) @ t
        # (3.21) \beta_{ML}^{-1} = \sum (t_n - w_{ML}^\top * \phi(x_n))^2 / N
        # (3.21') \beta_{ML}^{-1} = (t - \Phi * w_{ML})^\top * (t - \Phi * w_{ML}) / N
        # \beta is presision, so \beta^{-1} is variance
        self.var = np.mean(np.square(X @ self.w - t))

    def _predict(self, X, return_std=False):
        # (3.3') y(X, w) = \Phi * w
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
