import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier



class RandomForestClassifier :
    model = RandomForestClassifier(max_depth=5,n_estimators=100)



    def __init__(self, X_data_Train, Y_data_Train):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train


    def FitModel(self):
        self.model.fit(self.XdataTrain, self.YdataTrain)



    def TestModel(self, X_data_Test, Y_data_Test):
        prediction = self.model.predict(X_data_Test)
        print('Accuracy Of  Random Forest Classifier: ', str(metrics.accuracy_score(Y_data_Test, prediction)) +
         "\n__________________________________________________________________")