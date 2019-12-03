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
#import preprocessing as pre
#import PreProcessingBsmsm as pre
import PreProcessingFinal as pre
preprocessingVar = pre.PrepProcessing()
tobeTrained , tobeLabel = preprocessingVar.GetData()

LinearModel = LR.Linear_Regression(tobeTrained,tobeLabel)
LinearModel.FitModel()
LinearModel.TrainModel()



# svr
Svr = sv.SVR(tobeTrained,tobeLabel)
Svr.FitModel()
Svr.TrainModel()

# KNN model
KnnModel = Knn.Knn_Regression(tobeTrained,tobeLabel,10)
KnnModel.FitModel()
KnnModel.TrainModel()

# Lasso_Regression
LassoModel = LaR.Lasso_Regression(tobeTrained,tobeLabel)
LassoModel.FitModel()
LassoModel.TrainModel()

# Polynomial_Regression
PRModel = PR.Polynomial_Regression(tobeTrained,tobeLabel,3)
PRModel.FitModel()
PRModel.TrainModel()

# Ridge_Regression

RRModel = RR.Ridge_Regression(tobeTrained,tobeLabel)
RRModel.FitModel()
RRModel.TrainModel()
