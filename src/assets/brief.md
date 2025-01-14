## Assignment

It is efficient to put machine learning models and other data science techniques into classes so that we can reuse them later and change attributes without changing the code behind these models. Independent concepts can also be put into independent classes: for example, the functioning of a cross-validate class should not affect the functioning of a linear regression class. These concepts are also referred to as high cohesion and loose coupling.

<br />

1. Create a class called ErrorCalculator that has methods to compute the residuals, standardised residuals, Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Name these methods get_residuals, get_standardised_residuals, get_mse and get_rmse respectively. You can also have a method, error_summary that prints the average, minimum and maximum of the standardised residuals, as well as the MSE and RMSE.

    - The class should have the following parameters:
        - y: A 1D array of the target variable, size n_observations
        - y_pred: A 1D array of the predicted values of the target variable, size n_observations

<br />

2. Create a generic class called Plotter. This class should have a method, run_calculations, to calculate the residuals if they have not yet been calculated, and a method plot, which simply plots a histogram of the residuals.

    - As before, the class should have the following parameters:

        - y: A 1D array of the target variable, size n_observations
        - y_pred: A 1D array of the predicted values of the target variable, size n_observations

<br />

3. Now create two child classes, HistogramPlotter and ScatterPlotter, that both inherit from Plotter. As the name suggests, HistogramPlotter.plot() should return a histogram of the residuals, whereas ScatterPlotter.plot() should return scatterplots of the residual versus predicted values and the predicted versus observed values. (Why do this, you ask? Inheriting in this way makes it easier for future developers to know that the method plot will return a plot, regardless of the type of plot specified by the child class).

    Use the model you built in the Multivariate Regression project to predict log-transformed salary (log_salary). Also create a second multiple regression model which does not include yearsrank as a feature. Save these model instances as model1 and model2. Remember to scale (standardise) the features before modelling.

<br />

    In other words, you should have two models:

    **Model 1 Features:**
        - exprior, yearsworked, yearsrank, market, degree, otherqual, position, male, Field_dummyvariable1, Field_dummyvariable2, Field_dummyvariable3, yearsabs

    **Model 2 Features:**
        - exprior, yearsworked, market, degree, otherqual, position, male, Field_dummyvariable1, Field_dummyvariable2, Field_dummyvariable3, yearsabs

<br />

4. Use ErrorCalculator, HistogramPlotter and ScatterPlotter to get accuracy metrics and diagnostic plots for model 1 and model 2 (as defined in Q3). To keep the notebook neat, your classes should be imported from separate python files. How does the model accuracy and diagnostics for the two models compare? Which is the better regression model?

<br />

5. **(OPTIONAL)** Create a class called InfluenceCalculator that takes a model fit object as an argument and has methods for calculating and displaying leverage values and Cook’s distance. Call these methods leverage, cooks_distance. This is easiest to do using the package statsmodels to create the linear regression model object. Are there any observations that may be influencing the models?

<br />

**Tip:** These classes should be able to work with any regression problem. You can re-use these classes in upcoming assignments when inspecting models and you can also make classes in future that can work for both regression and classification problems.