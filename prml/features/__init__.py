# https://github.com/ctgk/PRML/blob/master/prml/features/__init__.py

from .gaussian import GaussianFeatures
from .polynomial import PolynomialFeatures
from .sigmoidal import SigmoidalFeatures


__all__ = [
    "GaussianFeatures",
    "PolynomialFeatures",
    "SigmoidalFeatures"
]