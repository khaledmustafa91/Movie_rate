import numpy as np
import pandas as pd
import sklearn.svm as ml
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class SVR:
    svr = ml.SVR(kernel='linear', C=1.0, epsilon=0.1)
    def __init__(self,toBeTrained,toBeLabel):
        '''self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test'''
        self.toBeTrained = toBeTrained
        self.toBeLabel = toBeLabel
    def train_model(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.svr, self.toBeTrained, self.toBeLabel, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
    def fitData(self):
        self.svr.fit(self.toBeTrained, self.toBeLabel)
    def Predict(self):
        y_predict = self.svr.predict(self.x_test)
        return y_predict

'''
readData = preprocessing.PrepProcessing()

toBeTrained , X_train,y_train,X_test = readData.GetData()

# Support vector machine regression
svr = ml.SVR(kernel='linear' , C=1.0 , epsilon=0.1 ).fit(np.array(X_train) , np.array(y_train))
SVRf = svr.predict(X_test)
#plot
plt.scatter(x=X_train , y= y_train)
plt.scatter(x=X_train, y= SVRf)
plt.title("Support Vector machine regression")
plt.xlabel('')
plt.ylabel('')
plt.show()
'''