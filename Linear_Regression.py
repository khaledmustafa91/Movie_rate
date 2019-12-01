import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics


class Linear_Regression:
    model = linear_model.LinearRegression()

    def __init__(self, X_data_Train, Y_data_Train, X_data_Test="", Y_data_Test=""):
        Y_data_Train = np.expand_dims(Y_data_Train, axis=1)
        Y_data_Test = np.expand_dims(Y_data_Test, axis=1)
        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train
        self.XdataTest = X_data_Test
        self.YdataTest = Y_data_Test

    def FitModel(self):
        self.model.fit(self.XdataTrain.T, self.YdataTrain)

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.model, self.XdataTrain.T, self.YdataTrain, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("Linear Regression Model " + "\nMSE: " + str(np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores)) + "\n__________________________________________________________________")
        # print("Scores: "+ str(scores))

    def TrainAndTestModel(self):
        prediction = self.model.predict(self.XdataTest)
        print("Linear Regression Model " + "\nMSE: " + str(metrics.mean_squared_error(self.YdataTest,
                                                                                      prediction)) + "\n__________________________________________________________________")
        # print("Scores: "+ str(scores))
