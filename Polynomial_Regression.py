import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

class Polynomial_Regression:
    model = linear_model.LinearRegression()

    def __init__(self, X_data_Train, Y_data_Train,degree):

        poly_features = PolynomialFeatures(degree=degree)
        poly_X_data = poly_features.fit_transform(X_data_Train.T)
        self.XdataTrain = poly_X_data
        self.YdataTrain = np.atleast_2d(Y_data_Train).T
        self.degree=degree


    def FitModel(self):
        self.model.fit(self.XdataTrain, self.YdataTrain)

    def TestModel(self, X_data_Test, Y_data_Test):
        Y_data_Test = np.atleast_2d(Y_data_Test).T
        poly_features = PolynomialFeatures(degree=self.degree)
        X_data_Test= poly_features.fit_transform(X_data_Test.T)

        prediction = self.model.predict(X_data_Test)
        print("Polynomial Regression Model " + "\nMSE: " + str(
            metrics.r2_score(Y_data_Test, prediction)) + "\nR2: " + str(
            metrics.mean_squared_error(Y_data_Test, prediction)) + "\n_____________________________________________")
