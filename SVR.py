import numpy as np
import pandas as pd
import sklearn.svm as ml
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics


class SVR:
    model = ml.SVR(kernel='linear', C=1.0, epsilon=0.1)

    def __init__(self, toBeTrained, toBeLabel):
        self.toBeTrained = toBeTrained
        self.toBeLabel = toBeLabel


    def FitModel(self):
        self.model.fit(self.toBeTrained.T, self.toBeLabel)

    def TestModel(self, X_data_Test, Y_data_Test):
        Y_data_Test = np.atleast_2d(Y_data_Test).T
        prediction = self.model.predict(np.array(X_data_Test).T)
        print("Support vector machine Regression Model " + "\nMSE: " + str(metrics.mean_squared_error( Y_data_Test,
                                                                                                       prediction)) + "\nR2: " + str(metrics.r2_score( Y_data_Test,
                                                                                                       prediction)) + "\n__________________________________________________________________")
