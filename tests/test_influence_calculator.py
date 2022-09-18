from utilities.influence_calculator import InfluenceCalculator
import statsmodels.api as sm
import numpy as np
import pytest


@pytest.mark.parametrize(
    ["X", "y", "cooks_d", "p_values"],
    [
        [
            [1,2,3,4],
            [5,6,7,8],
            np.array([7.13436385e-02, 8.87573964e-02, 1.35837018e-31, 1.22448980e+00]),
            np.array([0.80669949, 0.78520387, 1., 0.34924043]),
        ],
    ],
)
def test_cooks_distance(X, y, cooks_d, p_values):
    model = sm.OLS(y, X).fit()
    modelled_cooks_d = model.get_influence().cooks_distance[0]
    modelled_p_values = model.get_influence().cooks_distance[1]

    assert all(cooks_d.round(5) == modelled_cooks_d.round(5))
    assert all(p_values.round(5) == modelled_p_values.round(5))


@pytest.mark.parametrize(
    ["X", "y", "leverage"],
    [
        [
            [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20]],
            [5,6,7,8],
            np.array([0.7, 0.3, 0.3, 0.7]),
        ],
        [
            [[98, 45,765], [123,4567,234], [567,234,0]],
            [67,-21, 0],
            np.array([1., 1., 1.])
        ],
    ],
)
def test_leverage(X, y, leverage):
    model = sm.OLS(y, X).fit()
    modelled_leverage = model.get_influence().hat_matrix_diag

    assert all(leverage.round(5) == modelled_leverage.round(5))

