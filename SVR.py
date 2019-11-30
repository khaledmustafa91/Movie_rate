import numpy as np
import pandas as pd
import sklearn.svm as ml
import matplotlib.pyplot as plt
import preprocessing


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