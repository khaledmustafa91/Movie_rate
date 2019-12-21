import numpy as np
from sklearn import metrics
from sklearn import neighbors


class Knn_Regression:
    K=1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    def __init__(self, X_data_Train, Y_data_Train, KK=1):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train
        self.K=KK

    def FitModel(self):
        self.model = neighbors.KNeighborsRegressor(n_neighbors=self.K)
        self.model.fit(self.XdataTrain.T, self.YdataTrain)


    def TestModel(self,X_data_Test="", Y_data_Test=""):
        Y_data_Test = np.atleast_2d(Y_data_Test).T
        prediction = self.model.predict(np.array(X_data_Test).T)
        print("KNN Regression Model " + "\nK Value " + str(self.K) + "\nMSE: " + str(
            metrics.mean_squared_error(Y_data_Test, prediction))+ "\nR2: " + str(
            metrics.r2_score(Y_data_Test, prediction)) + "\n_____________________________________________")
