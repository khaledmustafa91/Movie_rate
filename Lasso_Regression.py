import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics


class Lasso_Regression:
    alpha = 0.2
    model = linear_model.Lasso(alpha=alpha)

    def __init__(self, X_data_Train, Y_data_Train, alpha=0.2):
        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train
        self.alpha = alpha

    def FitModel(self):
        self.model = linear_model.Ridge(alpha=self.alpha)
        self.model.fit(self.XdataTrain.T, self.YdataTrain)

    def TestModel(self, X_data_Test, Y_data_Test):
        Y_data_Test = np.atleast_2d(Y_data_Test).T
        prediction = self.model.predict(np.array(X_data_Test).T)
        print("Lasso Regression Model " + "\nMSE: " + str(
            metrics.r2_score(Y_data_Test, prediction)) + "\nR2: " + str(
            metrics.mean_squared_error(Y_data_Test, prediction)) + "\n_____________________________________________")
