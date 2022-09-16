from utilities.validator import check_for_array_validity
from collections import OrderedDict
import numpy as np
import json


class ErrorCalculator:
    def __init__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat
        self._check_parameters()

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

    def _check_parameters(self):
        self._check_types(self.y, "y")
        self._check_types(self.y_hat, "y_hat")
        self._check_dimensions(self.y, "y")
        self._check_dimensions(self.y_hat, "y_hat")
        self._check_for_equal_observation_count()
        self._check_for_more_than_two_entries(self.y, "y")
        self._check_for_more_than_two_entries(self.y_hat, "y_hat")

    def _check_types(self, parameter, name):
        if type(parameter) != np.ndarray:
            expected_type = "numpy array"
            message = self._parameter_type_error_message(expected_type, name)
            raise TypeError(message)

    def _check_dimensions(self, parameter, name):
        if parameter.ndim != 1:
            dimensions = "1 dimension"
            message = self._parameter_type_error_message(dimensions, name)
            raise TypeError(message)

    def _check_for_equal_observation_count(self):
        if self.y.size != self.y_hat.size:
            observations = "equal observations"
            variables = "y & y_hat"
            message = self._parameter_type_error_message(observations, variables)
            raise TypeError(message)

    def _check_for_more_than_two_entries(self, parameter, name):
        if parameter.size < 2:
            expectation = "more than 2 values"
            message = self._parameter_type_error_message(expectation, name)
            raise TypeError(message)

    @staticmethod
    def _parameter_type_error_message(expected, parameter):
        return f"expected a {expected} for the parameter {parameter}"
