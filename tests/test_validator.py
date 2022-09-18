from utilities.inference_calculator import InferenceCalculator
from utilities.error_calculator import ErrorCalculator
from utilities.plotter import Plotter
import statsmodels.api as sm
import numpy as np
import pytest


@pytest.mark.parametrize(
    ["Constructor", "y", "y_hat", "match"],
    [
        (ErrorCalculator, [0], np.array([1]), "^.* numpy array .* parameter y$"),
        (ErrorCalculator, np.array([2]), [3], "^.* numpy array .* parameter y_hat$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_not_np_ndarrays(Constructor, y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Constructor(y, y_hat)


@pytest.mark.parametrize(
    ["Constructor", "y", "y_hat", "match"],
    [
        (ErrorCalculator, np.array([[4], [5]]), np.array([6]), "^.* 1 dimension .* parameter y$"),
        (ErrorCalculator, np.array([7]), np.array([[8], [9]]), "^.* 1 dimension .* parameter y_hat$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_not_one_dimensional(Constructor, y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Constructor(y, y_hat)


@pytest.mark.parametrize(
    ["Constructor", "y", "y_hat", "match"],
    [
        (ErrorCalculator, np.array([]), np.array([]), "^.* more than 2 values .* parameter y & y_hat$"),
        (ErrorCalculator, np.array([1]), np.array([1]), "^.* more than 2 values .* parameter y & y_hat$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_empty(Constructor, y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Constructor(y, y_hat)


@pytest.mark.parametrize(
    ["Constructor", "y", "y_hat", "match"],
    [
        (
            ErrorCalculator,
            np.array([10, 11]),
            np.array([12]),
            "^.* equal observations .* parameter y & y_hat$",
        ),
        (
            ErrorCalculator,
            np.array([13]),
            np.array([14, 15]),
            "^.* equal observations .* parameter y & y_hat$",
        ),
    ],
)
def test_that_error_is_raised_when_parameter_sizes_are_not_equal(Constructor, y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Constructor(y, y_hat)


@pytest.mark.parametrize(
    ["algorithm"],
    [[sm.GLS], [sm.GLSAR], [sm.OLS], [sm.WLS]],
)
def test_that_fitted_least_squares_models_work(algorithm):
    X = [1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10]
    model = algorithm(y, X).fit()


@pytest.mark.parametrize(
    ["arg"],
    [
        [[1, 2, 3]],
        [{4, 5, 6}],
        [{7:8, 9:10, 11:12}],
        [(13, )],
    ],
)
def test_that_constructor_is_passed_a_fitted_model(arg):
    with pytest.raises(TypeError, match="^.* fitted model .* 'statsmodels'$"):
        InferenceCalculator(arg)
