import numpy as np
import pandas as pd
import sklearn.svm as ml
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics


class SVR:
    svr = ml.SVR(kernel='linear', C=1.0, epsilon=0.1)

    def __init__(self, toBeTrained, toBeLabel, x_test="", y_test=""):
        self.x_test = x_test
        self.y_test = y_test
        self.toBeTrained = toBeTrained
        self.toBeLabel = toBeLabel

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.svr, self.toBeTrained.T, self.toBeLabel, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("Support vector machine Regression Model " + "\nMSE: " + str(np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores)) + "\n__________________________________________________________________")

    def FitModel(self):
        self.svr.fit(self.toBeTrained.T, self.toBeLabel)

    def TrainAndTestModel(self):
        y_predict = self.svr.predict(self.x_test)
        print("Support vector machine Regression Model " + "\nMSE: " + str(metrics.mean_squared_error(self.y_test,
                                                                                                      y_predict)) + "\n__________________________________________________________________")
