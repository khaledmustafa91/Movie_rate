import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import neighbors



class Knn_Classifier:
    K=1
    model = neighbors.KNeighborsClassifier(n_neighbors=K)

    def __init__(self, X_data_Train, Y_data_Train, KK=1):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train

        self.K=KK

    def FitModel(self):
        self.model = neighbors.KNeighborsClassifier(n_neighbors=self.K)
        self.model.fit(self.XdataTrain, self.YdataTrain)




    def TestModel(self, X_data_Test, Y_data_Test):
        prediction = self.model.predict(X_data_Test)
        print('Accuracy Of Knn Classifier: ', str(metrics.accuracy_score(Y_data_Test, prediction)) +
         "\n__________________________________________________________________")