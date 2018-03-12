# https://github.com/ctgk/PRML/blob/master/prml/linear/__init__.py

from prml.linear.bayesian_regressor import BayesianRegressor, EmpiricalBayesRegressor
from prml.linear.linear_regressor import LinearRegressor
from prml.linear.ridge_regressor import RidgeRegressor
from prml.linear.least_squares_classifier import LeastSquaresClassifier
from prml.linear.logistic_regressor import LogisticRegressor
from prml.linear.softmax_regressor import SoftmaxRegressor

__all__ = [
    "BayesianRegressor",
    "EmpiricalBayesRegressor",
    "LinearRegressor",
    "LeastSquaresClassifier",
    "LogisticRegressor",
    "RidgeRegressor",
    "SoftmaxRegressor",
]