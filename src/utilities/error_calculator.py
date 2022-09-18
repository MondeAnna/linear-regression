from utilities.validator import check_for_array_validity
from collections import OrderedDict
import numpy as np
import json


class ErrorCalculator:
    def __init__(self, y, y_hat):
        check_for_array_validity(y, y_hat)
        self.y = y
        self.y_hat = y_hat

    def get_residuals(self):
        return self.y - self.y_hat

    def get_standardised_residuals(self):
        residuals = self.get_residuals()
        deviation = self._get_standard_residual_deviation()
        if (residuals == 0).all():
            return np.zeros(residuals.size)
        return residuals / deviation

    def get_mse(self):
        squared_residuals = self._get_squared_residuals()
        sum_of_squared_residuals = squared_residuals.sum()
        return sum_of_squared_residuals / squared_residuals.size

    def get_rmse(self):
        return np.sqrt(self.get_mse())

    def error_summary(self):
        standardised_residuals = self.get_standardised_residuals()
        summary = OrderedDict(
            {
                "Average Standardised Residuals": np.average(standardised_residuals),
                "Minimum Standardised Residuals": standardised_residuals.min(),
                "Maximum Standardised Residuals": standardised_residuals.max(),
                "MSE": self.get_mse(),
                "RMSE": self.get_rmse(),
            }
        )
        print(json.dumps(summary, indent=4), end="")

    def _get_standard_residual_deviation(self):
        squared_residuals = self._get_squared_residuals()
        numerator = squared_residuals.sum()
        denominator = squared_residuals.size - 2
        return np.sqrt(numerator / denominator)

    def _get_squared_residuals(self):
        return self.get_residuals() ** 2

