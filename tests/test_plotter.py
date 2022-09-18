from utilities.plotter import Plotter
from collections import OrderedDict
import numpy as np
import pytest


@pytest.mark.parametrize(
    ["y", "y_hat", "residuals", "std_residuals"],
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), np.zeros(3), np.zeros(3)),
        (
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
            np.full((3), -3),
            np.full(3, (-3 / np.sqrt(27))),
        ),
    ],
)
def test_calculation_of_residuals(y, y_hat, residuals, std_residuals):
    expected = OrderedDict(
        {"Residuals": residuals, "Standardised Residuals": std_residuals,}
    )

    plotter = Plotter(y, y_hat)
    result = plotter.run_calculations()

    for a, b in zip(expected.values(), result.values()):
        assert all(a == b)

