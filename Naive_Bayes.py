import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB



class Naive_Bayes :
    model = GaussianNB()



    def __init__(self, X_data_Train, Y_data_Train, X_data_Test="", Y_data_Test=""):

        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train
        self.XdataTest = X_data_Test
        self.YdataTest = Y_data_Test

    def FitModel(self):
        self.model.fit(self.XdataTrain, self.YdataTrain)

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.model, self.XdataTrain, self.YdataTrain, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("Decision Tree Classifier Model " + "\nK Value "+str(self.K) +"\nNumber of Parts Data: " + str(len(scores)) + "\nMSE: " + str(
            np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores)) + "\n__________________________________________________________________")

    def TestModel(self, XdataTest="", YdataTest=""):
        prediction = self.model.predict(self.XdataTest)
        print('Accuracy Of Naive Bayes Classifier : ', str(metrics.accuracy_score(self.YdataTest, prediction)) +
         "\n__________________________________________________________________")