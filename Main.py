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
#import preprocessingBsmsm as pre

preprocessingVar = pre.PrepProcessing()
tobeTrained , tobeLabel = preprocessingVar.GetData()

def main():
    print("If You Want to show our trail Press 1 \nIf You want to Test Press 2 ")
    ch=input()
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
    else:
        print("Enter X Data Name : ")
        xname=input()
        print("Enter Y Data Name : ")
        yname=input()
        XTest=pd.read_csv(xname)
        YTest = pd.read_csv(yname)

        LinearModel = LR.Linear_Regression(tobeTrained, tobeLabel,XTest,YTest)
        LinearModel.FitModel()
        LinearModel.TrainAndTestModel()

        # svr
        Svr = sv.SVR(tobeTrained, tobeLabel)
        Svr.FitModel()
        Svr.TrainAndTestModel()

        # KNN model
        KnnModel = Knn.Knn_Regression(tobeTrained, tobeLabel, 10,XTest,YTest)
        KnnModel.FitModel()
        KnnModel.TrainModel()

        # Lasso_Regression
        LassoModel = LaR.Lasso_Regression(tobeTrained, tobeLabel,XTest,YTest)
        LassoModel.FitModel()
        LassoModel.TrainAndTestModel()

        # Polynomial_Regression
        PRModel = PR.Polynomial_Regression(tobeTrained, tobeLabel, 3,XTest,YTest)
        PRModel.FitModel()
        PRModel.TrainAndTestModel()

        # Ridge_Regression

        RRModel = RR.Ridge_Regression(tobeTrained, tobeLabel,XTest,YTest)
        RRModel.FitModel()
        RRModel.TrainAndTestModel()

main()

