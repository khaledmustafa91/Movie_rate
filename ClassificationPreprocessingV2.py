import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import Logistic_Regression as LR
import SVM as svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
class Preprocessing:
    dataColumns = ["revenue", "vote_count", "runtime", "popularity", "budget", "genres", "keywords", "production_companies", "cast", "crew", "rate"]
    datatest = ["budget", "popularity", "revenue", "runtime", "vote_count"]
    Data = np.nan
    x = np.nan
    dataMissing = np.nan

    def __init__(self, datasetmovies, datasetcredits):
        self.datasetmovies = pd.read_csv(datasetmovies)
        self.datasetcredits = pd.read_csv(datasetcredits)
        self.datasetcredits = self.datasetcredits.drop("title", axis=1)
        self.Data = self.datasetcredits.set_index("movie_id").join(self.datasetmovies.set_index("id"), on="movie_id", sort=True)

    def deleteMissigData(self):
        self.Data.dropna(axis="index", how="any", thresh=None, subset=self.dataColumns, inplace=True)
        self.dataMissing = self.Data[self.Data.popularity == 0]
        self.dataMissing = self.Data[self.Data.revenue == 0]
        self.dataMissing = self.Data[self.Data.runtime == 0]
        self.dataMissing = self.Data[self.Data.vote_count == 0]
        self.dataMissing = self.Data[self.Data.budget == 0]
        self.Data = self.Data[self.Data.popularity != 0]
        self.Data = self.Data[self.Data.revenue != 0]
        self.Data = self.Data[self.Data.runtime != 0]
        self.Data = self.Data[self.Data.vote_count != 0]
        self.Data = self.Data[self.Data.budget != 0]


    def dropColumns(self):
        dropList = []
        for i in self.Data:
            if i not in self.dataColumns:
                dropList.append(i)
        self.Data.drop(dropList, axis="columns", inplace=True)

    def oneHotEncoding(self, data):
        aset = set()
        for i in data:
            for j in i:
                aset.add(j)
        print("the set     ", aset)
        arrayOfUniqueValues = []
        for i in aset:
            try:
                arrayOfUniqueValues.append(int(i))
            except:
                continue

        arrayOfUniqueValues = np.array(arrayOfUniqueValues)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(arrayOfUniqueValues)
        # print("integer_encoded",integer_encoded)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        #print("onehot_encoded",onehot_encoded)
        dicArray_Label = dict()
        for arr, lblenc in zip(arrayOfUniqueValues, integer_encoded):
            dicArray_Label[arr] = lblenc

        dicLabel_Bin = dict()
        for i in integer_encoded:
            v = onehot_encoder.transform(integer_encoded[i,:])
            dicLabel_Bin[integer_encoded[i]] = np.array(v)
            i += 1

        return dicArray_Label, dicLabel_Bin

    def mappingDataToOneHotEncoding(self,data,colname):
        dicarr_lbl, diclbl_bin = self.oneHotEncoding(data)
        retData = []
        for i in data:
            retData.append(diclbl_bin[dicarr_lbl[i]])

        df = pd.DataFrame(data=data, columns=[colname])

        print("data frame ", df)
        print("data before ", self.Data)
        print("data after ", self.Data)

        self.Data = pd.concat([self.Data,df],axis=1)


    def mapColumn(self,dataToMap, columnName):
        # [1] get all unique values
        # [2] create list of shape[(number of rows, number of unique value)]
        # [3] represent the column in the list

        # get unique values in aset
        aset = set()
        for i in dataToMap:
            for j in i:
                aset.add(j)

        # get new columns name in name list to add them in dataFrame each column =
        # oldDataColumnName + the Unique Value Of the new Column
        names = []
        for i in aset:
            names.append(columnName + i)

        # Empty dataList
        data = np.zeros([len(dataToMap), len(aset)])

        # new Dada Frame
        df = pd.DataFrame(data=data, columns=names)

        row = 0
        for i in dataToMap:
            for j in i:
                df[columnName + j][row] = 1
            row += 1
        #concatenate
        for i in names:
            self.Data[i] = np.array(df[i]).T

    def reformat(self):
        #get id from genres -> delete its Column and apply oneHotEncoing
        genresId = self.getIDFromJSON(np.copy(self.Data["genres"]), "id", False)
        self.Data.drop("genres", axis="columns", inplace=True)
        self.mapColumn(genresId, "genres")

        #self.Data.drop("cast", axis="columns", inplace=True)
        #self.Data.drop("crew", axis="columns", inplace=True)
        #        self.Data.drop("title", axis="columns", inplace=True)


        productionCompaniesId = self.getIDFromJSON(np.copy(self.Data["production_companies"]), "id", False)
        self.Data.drop("production_companies", axis="columns", inplace=True)
        self.mapColumn(productionCompaniesId, "production_companies")

        keywordsId = self.getIDFromJSON(np.copy(self.Data["keywords"]), "id", False)
        self.Data.drop("keywords", axis="columns", inplace=True)
        self.mapColumn(keywordsId, "keywords")

        self.x = pd.DataFrame(self.Data)
        self.x.drop( axis="columns", inplace=True,columns=["cast", "crew", "rate"])
        #print(self.x)

    def getIDFromJSON(self, arr, key, int_str):
        for i in range(len(arr)):
            str1 = arr[i].replace(']', '').replace('[', '')
            l = str1.replace('"', '').replace(',', '').replace('{', '').replace('}', '').replace(':', '')
            l = l.split(' ')
            listOfOneElement = []
            if key == 'Directing' or key == 'production':
                for j in range(len(l)):
                    if l[j] == 'id':
                        id = int(l[j + 1])
                        j += 1
                        while j < len(l):
                            if l[j] == key:
                                listOfOneElement.append(id)
                                break
                            if l[j] == "id":
                                id = int(l[j + 1])
                            j += 1
            else:
                for j in range(len(l)):
                    if l[j] == key:
                        try:
                            if int_str == True:
                                listOfOneElement.append(int(l[j + 1]))
                            else:
                                listOfOneElement.append(l[j + 1])
                        except:
                            continue

            arr[i] = listOfOneElement
        return arr

    def meanNormalization(self):
        for i in range(len(self.Data)):
            mx = max(self.data[i])
            mn = min(self.data[i])
            meu = np.mean(self.data[i])
            self.data[i] = (self.data[i] - meu) / (mx - mn)

    def fillMissingTestData(self, XdataTrain, test):
        model = NearestNeighbors(n_neighbors=1)
        model.fit(XdataTrain)
        print(np.shape(XdataTrain),np.shape([test]))
        kNearset = model.kneighbors([test], return_distance=False)
        print(kNearset[0][0])
        rowIndex = kNearset[0][0]
        print( XdataTrain[rowIndex,:])
        return 0

    def fillMissingTrainData(self, trainData):
        for i in range(np.shape(trainData)[1]):
            a = trainData[:, i]
            print("before mean " ,a)
            mean = np.median(a)
            trainData[:, i] = np.where(a == 0,mean,a)
            print("after mean ", trainData[:,i])
        return trainData

    def test(self):

        tobeTrained = []
        for i in self.datatest:
            tobeTrained.append(self.x[i])

        #for i in self.generes:
        #    tobeTrained.append(self.Data[i])
        tobeTrained = np.array(tobeTrained)
        tobeTrained = tobeTrained.T
        print(tobeTrained[:, 0])
        y = []
        for i in self.Data["rate"]:
            if i == "High":
                y.append(1)
            elif i == "Low":
                y.append(-1)
            else:
                y.append(0)
        #del y[0]

        X_train, X_test, y_train, y_test = train_test_split(tobeTrained, y, test_size=0.20)
        print("length of X_train  ", np.shape(X_train))
        X_trainWithoutPCA = X_train
        X_train, X_test = self.dimensionaltyReduction(5, X_train, X_test)



        obj1 = LR.Logistic_Regression(X_train, y_train, X_test, y_test)
        obj1.FitModel()
        obj1.TrainAndTestModel()

        obj2 = svm.SVM(X_train, y_train, X_test, y_test)
        obj2.FitModel()
        obj2.TrainAndTestModel()

        testaya = [0, 1.929883,  0, 90, 16]
        self.fillMissingTestData(X_trainWithoutPCA, testaya)


    def dimensionaltyReduction(self,numberOfPC,dataTrain,dataTest):
        dataTrans = StandardScaler().fit_transform(dataTrain)
        testdataTrans = StandardScaler().fit_transform(dataTest)
        pca = PCA(n_components=numberOfPC)
        principalComponents = pca.fit_transform(dataTrans)
        print(pca.explained_variance_ratio_)

        columnsNames= []
        for i in range(numberOfPC):
            columnsNames.append('PC'+str(i))

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=columnsNames)
        dataTest = pca.transform(testdataTrans)
        return principalDf , dataTest


obj = Preprocessing('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv')
obj.dropColumns()
obj.deleteMissigData()
obj.reformat()
obj.test()

'''import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import Logistic_Regression as LR
import SVM as svm


class Preprocessing:
    dataColumns = ["revenue", "vote_count", "runtime", "popularity", "genres", "keywords", "production_companies", "cast", "crew", "rate"]
    datatest = ["revenue", "vote_count", "runtime", "popularity"]
    Data = np.nan

    def __init__(self, datasetmovies, datasetcredits):
        self.datasetmovies = pd.read_csv(datasetmovies)
        self.datasetcredits = pd.read_csv(datasetcredits)
        self.datasetcredits = self.datasetcredits.drop("title", axis=1)
        self.Data = self.datasetcredits.set_index("movie_id").join(self.datasetmovies.set_index("id"), on="movie_id", sort=True)

    def deleteMissigData(self):
        self.Data.dropna(axis= "index", how="any", thresh=None, subset=self.dataColumns, inplace=True)

    def dropColumns(self):
        dropList = []
        for i in self.Data:
            if i not in self.dataColumns:
                dropList.append(i)
        self.Data.drop(dropList, axis="columns", inplace=True)

    def mapColumn(self,dataToMap, columnName):
        # [1] get all unique values
        # [2] create list of shape[(number of rows, number of unique value)]
        # [3] represent the column in the list

        # get unique values in aset
        aset = set()
        for i in dataToMap:
            for j in i:
                aset.add(j)

        # get new columns name in name list to add them in dataFrame each column =
        # oldDataColumnName + the Unique Value Of the new Column
        names = []
        for i in aset:
            names.append(columnName + i)

        # Empty dataList
        data = np.zeros([len(dataToMap), len(aset)])

        # new Dada Frame
        df = pd.DataFrame(data=data, columns=names)
        row = 0
        for i in dataToMap:
            for j in i:
                df[columnName + j][row] = 1
            row += 1
        #concatenate
        for i in names:
            self.Data[i] = np.array(df[i]).T

    def reformat(self):
        #get id from genres -> delete its Column and apply oneHotEncoing

        genresId = self.getIDFromJSON(np.copy(self.Data["genres"]), "id", False)
        self.Data.drop("genres", axis="columns", inplace=True)
        self.mapColumn(genresId, "genres")


        productionCompaniesId = self.getIDFromJSON(np.copy(self.Data["production_companies"]), "id", False)
        self.Data.drop("production_companies", axis="columns", inplace=True)
        self.mapColumn(productionCompaniesId, "production_companies")

        keywordsId = self.getIDFromJSON(np.copy(self.Data["keywords"]), "id", False)
        self.Data.drop("keywords", axis="columns", inplace=True)
        self.mapColumn(keywordsId, "keywords")

    def getIDFromJSON(self, arr, key, int_str):
        for i in range(len(arr)):
            str1 = arr[i].replace(']', '').replace('[', '')
            l = str1.replace('"', '').replace(',', '').replace('{', '').replace('}', '').replace(':', '')
            l = l.split(' ')
            listOfOneElement = []
            if key == 'Directing' or key == 'production':
                for j in range(len(l)):
                    if (l[j] == 'id'):
                        id = int(l[j + 1])
                        j += 1
                        while j < len(l):
                            if l[j] == key:
                                listOfOneElement.append(id)
                                break
                            if l[j] == "id":
                                id = int(l[j + 1])
                            j += 1
            else:
                for j in range(len(l)):
                    if l[j] == key:
                        try:
                            if int_str == True:
                                listOfOneElement.append(int(l[j + 1]))
                            else:
                                listOfOneElement.append(l[j + 1])
                        except:
                            continue

            arr[i] = listOfOneElement
        return arr

    def meanNormalization(self):
        for i in range(len(self.Data)):
            mx = max(self.data[i])
            mn = min(self.data[i])
            meu = np.mean(self.data[i])
            self.data[i] = (self.data[i] - meu) / (mx - mn)

    def test(self):
        tobeTrained = []
        for i in self.datatest:
            tobeTrained.append(self.Data[i])

        #for i in self.generes:
        #    tobeTrained.append(self.Data[i])
        tobeTrained = np.array(tobeTrained)

        y = []
        for i in self.Data["rate"]:
            if i == "High":
                y.append(1)
            elif i == "Low":
                y.append(-1)
            else:
                y.append(0)
        #del y[0]
        print("to be trained dim ",np.shape(tobeTrained),"\n y dim ", np.shape(y))
        X_train, X_test, y_train, y_test = train_test_split(tobeTrained.T, y, test_size=0.20)


        obj1 = LR.Logistic_Regression(X_train, y_train, X_test, y_test)
        obj1.FitModel()
        obj1.TrainAndTestModel()

        obj2 = svm.SVM(X_train, y_train, X_test, y_test)
        obj2.FitModel()
        obj2.TrainAndTestModel()


obj = Preprocessing('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv')
obj.dropColumns()
obj.deleteMissigData()
obj.reformat()
obj.test()'''

