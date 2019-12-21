import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import sklearn.svm as ml
import Linear_Regression as LR
import SVR as sv
import Knn_Regression as Knn
import Lasso_Regression as LaR
import Polynomial_Regression as PR
import Ridge_Regression as RR
import PreProcessingFinal as pre

import DecisionTreeClassifier as DTC
import Knn_Classifier as KNNC
import Naive_Bayes as NB
import RandomForestClassifier as RFC
import SVM as svm

def main():
    preprocessingVar = pre.PrepProcessing('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv')
    preprocessingVar.reformat()
    preprocessingVar.deleteMissingData()
    preprocessingVar.meanNormalization()
    tobeTrained, tobeLabel = preprocessingVar.GetData()

    print("If You Want to show our train Press 1 \nIf You want to Test Press 2 ")
    ch = input()
    if ch=='1':
        LinearModel = LR.Linear_Regression(tobeTrained, tobeLabel)
        LinearModel.FitModel()
        LinearModel.TrainModel()

        # svr
        Svr = sv.SVR(tobeTrained, tobeLabel)
        Svr.FitModel()
        Svr.TrainModel()

        # KNN model
        KnnModel = Knn.Knn_Regression(tobeTrained, tobeLabel, 5)
        KnnModel.FitModel()
        KnnModel.TrainModel()

        # Lasso_Regression
        LassoModel = LaR.Lasso_Regression(tobeTrained, tobeLabel)
        LassoModel.FitModel()
        LassoModel.TrainModel()

        # Polynomial_Regression
        PRModel = PR.Polynomial_Regression(tobeTrained, tobeLabel, 3)
        PRModel.FitModel()
        PRModel.TrainModel()

        # Ridge_Regression

        RRModel = RR.Ridge_Regression(tobeTrained, tobeLabel)
        RRModel.FitModel()
        RRModel.TrainModel()



        DecTree = DTC.DecisionTreeClassifier(tobeTrained,tobeLabel)
        DecTree.FitModel()
        DecTree.TrainModel()

    else:
        preprocessingVar1 = pre.PrepProcessing('tmdb_5000_movies_train.csv', 'tmdb_5000_credits_train.csv')
        preprocessingVar1.reformat()
        preprocessingVar1.deleteMissingData()
        preprocessingVar1.meanNormalization()

        XTest, YTest = preprocessingVar1.GetData()
        YTest = np.atleast_2d(YTest).T

        LinearModel = LR.Linear_Regression(tobeTrained, tobeLabel, XTest, YTest)
        LinearModel.FitModel()
        LinearModel.TrainAndTestModel()
        # svr
        '''
        Svr = sv.SVR(tobeTrained, tobeLabel)
        Svr.FitModel()
        Svr.TrainAndTestModel()
       '''
        # KNN model
        KnnModel = Knn.Knn_Regression(tobeTrained, tobeLabel, 10,XTest,YTest)
        KnnModel.FitModel()
        KnnModel.TrainModel()
        '''
        # Lasso_Regression
        LassoModel = LaR.Lasso_Regression(tobeTrained, tobeLabel,XTest,YTest)
        LassoModel.FitModel()
        LassoModel.TrainAndTestModel()
       '''
        # Polynomial_Regression
        PRModel = PR.Polynomial_Regression(tobeTrained, tobeLabel, 3, XTest, YTest)
        PRModel.FitModel()
        PRModel.TrainAndTestModel()

        # Ridge_Regression

        RRModel = RR.Ridge_Regression(tobeTrained, tobeLabel,XTest,YTest)
        RRModel.FitModel()
        RRModel.TrainAndTestModel()

main()

