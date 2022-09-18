from utilities.validator import check_for_array_validity
from utilities.error_calculator import ErrorCalculator
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:

    def __init__(self, y, y_hat):
        check_for_array_validity(y, y_hat)
        self.y = y
        self.y_hat = y_hat
        self.calculations = self.run_calculations()

    def run_calculations(self):
        calc = ErrorCalculator(self.y, self.y_hat)
        return OrderedDict(
            {
                "Residuals": calc.get_residuals(),
                "Standardized Residuals": calc.get_standardised_residuals(),
            }
        )

    def plot(self, model_name=""):
        title="Residuals Histogram"
        ylabel = "Count"

        residuals, std_residuals = self.calculations.keys()
        self._set_graph_globals()

        fig, ax = plt.subplots(1, 2)

        fig.suptitle(f"{model_name}\n{title}")

        sns.histplot(
            data=self.calculations[residuals],
            ax=ax[0]
        ).set(
            ylabel=ylabel,
            xlabel=residuals,
        )

        sns.histplot(
            data=self.calculations[std_residuals],
            ax=ax[1]
        ).set(
            ylabel=ylabel,
            xlabel=std_residuals,
        )

        plt.tight_layout()
        plt.show()

    def _set_graph_globals(self):
        sns.set(rc={
        'axes.labelsize': 18,
        'axes.labelweight': 'bold',
        'figure.figsize': (12, 8),
        'figure.titlesize': 36,
        'figure.titleweight': 'bold',
        'font.weight': 'bold',
    })


class HistogramPlotter(Plotter):
    ...


class ScatterPlotter(Plotter):

    def plot(self, model_name=""):
        xlabel="Predictions"
        title_a="Predictions vs. Residuals"
        title_b = "Scatter Plot"

        residuals, std_residuals = self.calculations.keys()
        super()._set_graph_globals()

        fig, ax = plt.subplots(1, 2)

        fig.suptitle(f"{model_name}\n{title_a}\n{title_b}")

        sns.scatterplot(
            x=self.y_hat,
            y=self.calculations[residuals],
            ax=ax[0]
        ).set(
            ylabel=residuals,
            xlabel=xlabel,
        )

        sns.scatterplot(
            x=self.y_hat,
            y=self.calculations[std_residuals],
            ax=ax[1]
        ).set(
            ylabel=std_residuals,
            xlabel=xlabel,
        )

        plt.tight_layout()
        plt.show()
