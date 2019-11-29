import numpy as np
import pandas as pd
import sklearn.svm as ml
import matplotlib as plt

readData = pd.read_csv()



# Support vector machine regression
svr = ml.SVR(kernel='linear' , C=1.0 , epsilon=0.1 ).fit(np.array('''features''') , np.array(''''target'''))
SVRf = svr.predict()
#plot
plt.scatter(x='''data1''' , y= '''target1 ''')
plt.scatter(x='''data1 ''' , y= '''SVRf''')
plt.title("Support Vector machine regression")
plt.xlabel('')
plt.ylabel('')
plt.show()