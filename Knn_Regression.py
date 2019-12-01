import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import neighbors


class Knn_Regression:
    K=1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)

    def __init__(self, X_data_Train, Y_data_Train, KK=1, X_data_Test="", Y_data_Test=""):
        Y_data_Train = np.expand_dims(Y_data_Train, axis=1)
        Y_data_Test = np.expand_dims(Y_data_Test, axis=1)
        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train
        self.XdataTest = X_data_Test
        self.YdataTest = Y_data_Test
        self.K=KK

    def FitModel(self):
        self.model = neighbors.KNeighborsRegressor(n_neighbors=self.K)
        self.model.fit(self.XdataTrain.T, self.YdataTrain)

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.model, self.XdataTrain.T, self.YdataTrain, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("KNN Regression Model " + "\nK Value "+str(self.K) +"\nNumber of Parts Data: " + str(len(scores)) + "\nMSE: " + str(
            np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores)) + "\n__________________________________________________________________")
        # print("Scores: "+ str(scores))

    def TrainAndTestModel(self):
        prediction = self.model.predict(self.XdataTest)
        print("KNN Regression Model " +"\nK Value "+str(self.K) + "\nMSE: " + str(
            metrics.mean_squared_error(self.YdataTest, prediction)) + "\n_____________________________________________")
