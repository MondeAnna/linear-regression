from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.graphics.regressionplots import influence_plot
from utilities.validator import check_model_validity
import matplotlib.pyplot as plt


class InfluenceCalculator:

    def __init__(self, model):
        check_model_validity(model)
        self.model = model

    def cooks_distance(self):
        return self.model.get_influence().cooks_distance

    def leverage(self):
        return self.model.get_influence().hat_matrix_diag

    def show(self, model_name):
        influence_plot(self.model, size=18, plot_alpha=0.625)
        leverage = "Leverage Graph"
        plt.title(f"{model_name}\n{leverage}", fontweight="bold", fontsize=36)
        plt.show()
