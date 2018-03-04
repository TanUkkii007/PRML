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
        # after observing N inputs
        # (3.51) S_N^{-1} = S_0^{-1} + \beta * \Phi^\top * \Phi
        # after observing another input
        # (3.51') S_{N+1}^{-1} = S_N^{-1} + \beta \phi(x_{N+1})\phi(x_{N+1})^\top
        # See problem 3.8
        w_precision = precision_prev + self.beta * X.T @ X
        # after observing N inputs
        # (3.50) m_N = S_N(S_0^{-1} * m_0 + \beta * \Phi^\top * t) 
        # after observing another input
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


class EmpricalBayesRegressor(Regressor):

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def _fit(self, X, t, max_iter=100):
        M = X.T @ X
        # np.linalg.eigvalsh
        # Compute the eigenvalues of a Hermitian or real symmetric matrix.
        # Returns: The eigenvalues in ascending order, each repeated according to
        # its multiplicity.
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]

            # (3.81) A = \alpha I + \beta \Phi^\top \Phi
            w_precision = self.alpha * eye + self.beta * X.T @
            # (3.84) m_N = \beta A^{-1} \Phi^\top t
            # m_N can be obtained by solving A m_N = \beta \Phi^\top t
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)

            # (3.91) \gamma = \sum_i \frac{\lambda_i}{\alpha + \lambda_i}
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))

            # updating \alpha and \beta

            # (3.92) \alpha = \frac{\gamma}{m_N^\top m_N}
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            # (3.95) \frac{1}{\beta} = \frac{1}{N - \gamma}\sum_n \{ t_n - m_N^\top \Phi_n \}^2
            self.beta = float(
                (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )

            if np.allclose(params, [self.alpha, self.beta]):
                break

        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)

    def log_evidence(self, X, t):
        """
        log evidence function
        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data
        Returns
        -------
        output : float
            log evidence
        """

        # np.linalg.slogdet
        # Compute the sign and (natural) logarithm of the determinant of an array.
        # Returns:
        #   sign : A number representing the sign of the determinant.
        #   logdet : The natural log of the absolute value of the determinant.

        # (3.86) \ln p(t|\alpha, \beta) =
        # M/2\ln\alpha
        # + N/2 \ln\beta
        # - E(m_N)
        # - 1/2\ln|A|
        # - N/2 \ln(2\pi)
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - self.beta * np.square(t - X @ self.w_mean).sum()
            - self.alpha * np.sum(self.w_mean ** 2)
            - np.linalg.slogdet(self.w_precision)[1]
            - len(t) * np.log(2 * np.pi)
        )

    def _predict(self, X, return_std=False, sample_size=None):
        if isinstance(sample_size, int):
            # predictive distribution
            # (3.58) p(t|x, t, \alpha, \beta) = N(t| m_N^\top * \phi(x), \sigma_N(x)^2)
            w_sample = np.random.multivariate_normal(self.w_mean, self.w_cov, size=sample_size)
            y = X @ w_sample.T
            return y

        y = X @ self.w_mean
        if return_std:
            # (3.59) \sigma_N(x)^2 = 1/\beta + \phi(x)^\top * S_N * \phi(x)
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y