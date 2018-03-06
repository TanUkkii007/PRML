# https://github.com/ctgk/PRML/blob/master/prml/linear/__init__.py

from prml.linear.bayesian_regressor import BayesianRegressor, EmpiricalBayesRegressor
from prml.linear.linear_regressor import LinearRegressor
from prml.linear.ridge_regressor import RidgeRegressor

__all__ = [
    "BayesianRegressor",
    "EmpiricalBayesRegressor",
    "LinearRegressor",
    "RidgeRegressor",
]