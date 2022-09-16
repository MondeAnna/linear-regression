from utilities.error_calculator import ErrorCalculator as Calc
from collections import OrderedDict
import numpy as np
import pytest
import json


@pytest.mark.parametrize(
    ["y", "y_hat", "expected"],
    [
        (np.array([16, 17, 18]), np.array([19, 20, 21]), np.array([-3, -3, -3])),
        (np.array([22, 23, 24]), np.array([27, 18, 224]), np.array([-5, +5, -200])),
        (np.array([25, 26, 27]), np.array([25, 26, 27]), np.array([0, 0, 0])),
    ],
)
def test_for_residuals(y, y_hat, expected):
    calc = Calc(y, y_hat)
    residuals = calc.get_residuals()
    assert all(expected == residuals)


@pytest.mark.parametrize(
    ["y", "y_hat", "expected"],
    [
        (np.array([39, 40, 41]), np.array([39, 40, 41]), np.array([0, 0, 0])),
        (
            np.array([42, 43, 44, 45]),
            np.array([132, 233, 634, 235]),
            np.array([-0.19446, -0.41053, -1.27480, -0.41053]),
        ),
        (
            np.array([-46, -47, -48]),
            np.array([135, 136, 137]),
            np.array([-0.57102, -0.57733, -0.58364]),
        ),
    ],
)
def test_for_standardised_residuals(y, y_hat, expected):
    calc = Calc(y, y_hat)
    standardised_residuals = calc.get_standardised_residuals()
    rounded_standardised_residuals = standardised_residuals.round(decimals=5)
    assert all(expected == rounded_standardised_residuals)


@pytest.mark.parametrize(
    ["y", "y_hat", "expected"],
    [
        (np.array([28, 29, 30]), np.array([28, 29, 30]), 0),
        (np.array([31, 32, 33, 34]), np.array([32, 33, 34, 35]), np.sqrt(2)),
        (np.array([35, 36, 37, 38]), np.array([135, 136, 137, 138]), np.sqrt(20_000)),
    ],
)
def test_for_standard_residual_deviation(y, y_hat, expected):
    calc = Calc(y, y_hat)
    standardised_residual_deviation = calc._get_standard_residual_deviation()
    assert expected == standardised_residual_deviation


@pytest.mark.parametrize(
    ["y", "y_hat", "expected"],
    [
        (np.array([-28, -29, -30]), np.array([-28, -29, -30]), 0),
        (np.array([50, 51, 52]), np.array([214, 326, 437]), 83_582),
        (np.array([-53, -54, -55]), np.array([548, 659, 760]), 511_265),
    ],
)
def test_for_mean_sqaure_error(y, y_hat, expected):
    calc = Calc(y, y_hat)
    mse = calc.get_mse()
    assert expected == mse


@pytest.mark.parametrize(
    ["y", "y_hat", "expected"],
    [
        (np.array([-45, 0, 300]), np.array([-45, 0, 300]), 0),
        (np.array([50, 0, 4, 8]), np.array([54, 0, 4, 8]), 2),
        (np.array([15, 20, 25, 30]), np.array([20, 25, 30, 35]), 5),
        (
            np.array([-926, 44, 565, 0]),
            np.array([56, -69, 7660, 0]),
            np.sqrt(12_829_029.5),
        ),
    ],
)
def test_for_root_mean_sqaure_error(y, y_hat, expected):
    calc = Calc(y, y_hat)
    rmse = calc.get_rmse()
    assert np.round(expected, decimals=5) == np.round(rmse, decimals=5)


def test_error_summary(capsys):
    summary = OrderedDict(
        {
            "Average Standardised Residuals": 0.35355339059327373,
            "Minimum Standardised Residuals": 0.0,
            "Maximum Standardised Residuals": 1.414213562373095,
            "MSE": 1.0,
            "RMSE": 1.0,
        }
    )
    expected = json.dumps(summary, indent=4)

    y = np.array([1, 2, 3, 4])
    y_hat = np.array([-1, 2, 3, 4])

    calc = Calc(y, y_hat)
    calc.error_summary()

    stdout, _ = capsys.readouterr()
    assert expected == stdout


@pytest.mark.parametrize(
    ["y", "y_hat", "match"],
    [
        ([0], np.array([1]), "^.* numpy array .* parameter y$"),
        (np.array([2]), [3], "^.* numpy array .* parameter y_hat$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_not_np_ndarrays(y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Calc(y, y_hat)


@pytest.mark.parametrize(
    ["y", "y_hat", "match"],
    [
        (np.array([[4], [5]]), np.array([6]), "^.* 1 dimension .* parameter y$"),
        (np.array([7]), np.array([[8], [9]]), "^.* 1 dimension .* parameter y_hat$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_not_one_dimensional(y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Calc(y, y_hat)


@pytest.mark.parametrize(
    ["y", "y_hat", "match"],
    [
        (np.array([]), np.array([]), "^.* more than 2 values .* parameter y$"),
        (np.array([1]), np.array([1]), "^.* more than 2 values .* parameter y$"),
    ],
)
def test_that_error_is_raised_if_parameters_are_empty(y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Calc(y, y_hat)


@pytest.mark.parametrize(
    ["y", "y_hat", "match"],
    [
        (
            np.array([10, 11]),
            np.array([12]),
            "^.* equal observations .* parameter y & y_hat$",
        ),
        (
            np.array([13]),
            np.array([14, 15]),
            "^.* equal observations .* parameter y & y_hat$",
        ),
    ],
)
def test_that_error_is_raised_when_parameter_sizes_are_not_equal(y, y_hat, match):
    with pytest.raises(TypeError, match=match):
        Calc(y, y_hat)
