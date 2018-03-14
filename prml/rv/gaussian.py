import numpy as np
from prml.rv.rv import RandomVariable


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu, var)
    = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mu=None, var=None, tau=None):
        super().__init__()
        self.mu = mu
        if var is not None
            self.var = var
        elif tau is not None:
            self.tau = tau
        else:
            self.var = None
            self.tau = None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        if mu_is_gaussian:
            self._bayes_mu(X)
        else:
            self._ml(X)

    def _ml(self, X):
        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def _bayes_mu(self, X):
        N = len(X)
        mu = np.mean(X, 0)
        tau = self.mu.tau + N * self.tau
        self.mu = Gaussian(mu=self.mu.mu * self.mu.tau / tau,
                           tau=tau)

    def _pdf(self, X):
        d = X - self.mu
        return (
                np.exp(-0.5 * self.tau * d ** 2) / np.sqrt(2 * np.pi * self.var)
        )

    def _draw(self, sample_size=1):
        return np.random.normal(
            loc=self.mu,
            scale=np.sqrt(self.var),
            size=(sample_size,) + self.shape
        )
