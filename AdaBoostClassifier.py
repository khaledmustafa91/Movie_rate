import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier




class AdaBoostClassifier:
    model  =AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",
                         n_estimators=800)

    def __init__(self, X_data_Train, Y_data_Train):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train


    def FitModel(self):
        self.model.fit(self.XdataTrain, self.YdataTrain)




    def TestModel(self, X_data_Test, Y_data_Test):
        prediction = self.model.predict(X_data_Test)
        print('Accuracy Of  AdaBoost Classifier: ', str(metrics.accuracy_score(Y_data_Test, prediction)) +
         "\n__________________________________________________________________")