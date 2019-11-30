import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

class Polynomial_Regression:
    model = linear_model.LinearRegression()

    def __init__(self, X_data, Y_data):
        Y_data = np.expand_dims(Y_data, axis=2)
        poly_features = PolynomialFeatures(degree=1)
        poly_X_data = poly_features.fit_transform(X_data.T)
        self.Xdata = poly_X_data
        self.Ydata = Y_data

    def FitModel(self):
        self.model.fit(self.Xdata, self.Ydata)

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.model, self.Xdata, self.Ydata, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("Polynomial Linear Regression" + "\nNumber of Parts Data: " + str(len(scores)) + "\nMSE: " + str(
            np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores)) + "\n__________________________________________________________________")
        #print("Scores: "+ str(scores))
