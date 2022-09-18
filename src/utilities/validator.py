from statsmodels.regression.linear_model import RegressionResultsWrapper
import numpy as np


def check_model_validity(model):
    if type(model) != RegressionResultsWrapper:
        raise TypeError("expected a fitted model rendered from 'statsmodels'")


def check_for_array_validity(y, y_hat):
    _check_array_type(y, "y")
    _check_array_type(y_hat, "y_hat")
    _check_dimensions(y, "y")
    _check_dimensions(y_hat, "y_hat")
    _check_for_equal_observation_count(y, y_hat)
    _check_for_more_than_two_entries(y, y_hat)


def _check_array_type(parameter, name):
    if type(parameter) != np.ndarray:
        expected_type = "numpy array"
        message = _parameter_type_error_message(expected_type, name)
        raise TypeError(message)


def _check_dimensions(parameter, name):
    if parameter.ndim != 1:
        dimensions = "1 dimension"
        message = _parameter_type_error_message(dimensions, name)
        raise TypeError(message)


def _check_for_equal_observation_count(y, y_hat):
    if y.size != y_hat.size:
        observations = "equal observations"
        variables = "y & y_hat"
        message = _parameter_type_error_message(observations, variables)
        raise TypeError(message)


def _check_for_more_than_two_entries(y, y_hat):
    if y.size < 2 or y_hat.size < 2:
        expectation = "more than 2 values"
        variables = "y & y_hat"
        message = _parameter_type_error_message(expectation, variables)
        raise TypeError(message)


def _parameter_type_error_message(expected, parameter):
    return f"expected {expected} for the parameter {parameter}"

