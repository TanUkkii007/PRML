# https://github.com/ctgk/PRML/blob/master/prml/linear/__init__.py

from prml.linear.bayesian_regressor import BayesianRegressor, EmpiricalBayesRegressor
from prml.linear.linear_regressor import LinearRegressor
from prml.linear.ridge_regressor import RidgeRegressor
from prml.linear.least_squares_classifier import LeastSquaresClassifier

__all__ = [
    "BayesianRegressor",
    "EmpiricalBayesRegressor",
    "LinearRegressor",
    "LeastSquaresClassifier",
    "RidgeRegressor",
]