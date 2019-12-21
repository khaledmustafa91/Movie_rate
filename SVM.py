import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier


class SVM:
    model = SVC(gamma='auto', C=1)
    model1=OneVsRestClassifier(model)
    def __init__(self, X_data_Train, Y_data_Train):
        self.XdataTrain = X_data_Train
        self.YdataTrain = Y_data_Train

    def FitModel(self):
        self.model=SVC(gamma='auto', C=3)
        self.model.fit(self.XdataTrain, self.YdataTrain)

        self.model1 = OneVsRestClassifier(SVC(gamma='auto', C=3))
        self.model1.fit(self.XdataTrain, self.YdataTrain)


    def TestModel(self, X_data_Test="", Y_data_Test=""):
        prediction = self.model1.predict(X_data_Test)
        print('Accuracy Of One VS Rest SVM : ', str(metrics.accuracy_score( Y_data_Test, prediction))+
              "\n__________________________________________________________________")

        prediction1 = self.model.predict(X_data_Test)
        print('Accuracy Of One VS One SVM : ', str(metrics.accuracy_score(Y_data_Test, prediction1))+
              "\n__________________________________________________________________")



