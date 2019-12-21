import numpy as np
from sklearn import linear_model
from sklearn import metrics


class Linear_Regression:
    model = linear_model.LinearRegression()

    def __init__(self, X_data_Train, Y_data_Train):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train

    def FitModel(self):
        self.model.fit(self.XdataTrain.T, self.YdataTrain)



    def TestModel(self, X_data_Test="", Y_data_Test=""):
        Y_data_Test = np.atleast_2d(Y_data_Test).T
        prediction = self.model.predict( X_data_Test.T)
        print("Linear Regression Model " + "\nMSE: " + str(metrics.mean_squared_error( Y_data_Test,                                    prediction)) + "\nR2: " + str(metrics.r2_score( Y_data_Test,
                                                  prediction)) + "\n__________________________________________________________________")
