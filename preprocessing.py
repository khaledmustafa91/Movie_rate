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

class PrepProcessing:
    dicOfLanguages = list()
    datasetcredits = np.nan
    data = []
    help = list()

    def __init__(self):
        self.datasetmovies = pd.read_csv('tmdb_5000_movies_train.csv')
        self.datasetcredits = pd.read_csv('tmdb_5000_credits_train.csv')

    def basic_step_of_preprocessing(self, str, datasetName):
        arr = []
        if datasetName == "datasetmovies":
            for x in self.datasetmovies[str]:
                arr.append(x)
        else:
            for x in self.datasetcredits[str]:
                arr.append(x)

        return np.array(arr)

    def transform_map(self, arr):
        aset = set()
        dic = dict()
        for i in range(len(arr)):
            aset.add(arr[i])
        counter = 1
        for setElement in aset:
            dic[setElement] = counter
            counter += 1
        return dic

    def mapping(self, dic, arr):
        for i in range(len(arr)):
            arr[i] = dic[arr[i]]
        return arr

    def getIDFromJSON(self, arr, key, int_str):
        for i in range(len(arr)):
            str1 = arr[i].replace(']', '').replace('[', '')
            l = str1.replace('"', '').replace(',', '').replace('{', '').replace('}', '').replace(':', '')
            l = l.split(' ')
            # print(l)
            listOfOneElement = []
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

    def map_spoken_language(self, spokenLanguages):
        for i in range(len(spokenLanguages)):
            for j in range(len(spokenLanguages[i])):
                if spokenLanguages[i][j] in self.dicOfLanguages:
                    spokenLanguages[i][j] = self.dicOfLanguages[spokenLanguages[i][j]]
                else:
                    self.dicOfLanguages[spokenLanguages[i][j]] = len(self.dicOfLanguages) + 1
                    spokenLanguages[i][j] = self.dicOfLanguages[spokenLanguages[i][j]]
        return spokenLanguages

    def reformat(self):
        self.datasetmovies = self.datasetmovies[0:3799]
        self.datasetcredits = self.datasetcredits[0:3799]
        self.help.append("id")
        self.data.append(self.basic_step_of_preprocessing('id', "datasetmovies"))

        self.help.append("budget")
        self.data.append(self.basic_step_of_preprocessing('budget', "datasetmovies"))

        self.help.append("revenue")
        self.data.append(self.basic_step_of_preprocessing('revenue', "datasetmovies"))

        self.help.append("original_language")
        org_lang = self.basic_step_of_preprocessing('original_language', "datasetmovies")
        self.dicOfLanguages = self.transform_map(org_lang)
        org_lang = self.mapping(self.dicOfLanguages, org_lang)
        self.data.append(org_lang)

        self.help.append("vote_count")
        voteCount = self.basic_step_of_preprocessing('vote_count', "datasetmovies")
        self.data.append(voteCount)

        self.help.append("vote_average")
        voteAvrg = self.basic_step_of_preprocessing('vote_average', "datasetmovies")
        self.data.append(voteAvrg)

        self.help.append("runtime")
        self.data.append(self.basic_step_of_preprocessing('runtime', "datasetmovies"))

        self.help.append("popularity")
        self.data.append(self.basic_step_of_preprocessing('popularity', "datasetmovies"))

        self.help.append("genres")
        genres = self.datasetmovies['genres']
        genres = self.getIDFromJSON(genres[:], 'id', True)
        self.data.append(genres)

        self.help.append("keywords")
        keywords = self.datasetmovies['keywords']
        keywords = keywords[:3799]
        keywords = self.getIDFromJSON(keywords, 'id', True)
        self.data.append(keywords)

        # Production Countries Pre-Processing
        self.help.append("production_countries")
        productionCountries = self.datasetmovies['production_countries']
        productionCountries = productionCountries[:3799]
        productionCountries = self.getIDFromJSON(productionCountries, 'id', True)
        # print(productionCountries)
        dicOfPrductioCountries = self.handlingCatigorialVariables(productionCountries, voteAvrg, voteCount, 1)
        productionCountries = self.putDictToList(dicOfPrductioCountries, productionCountries)
        self.data.append(productionCountries)

        # Production companies Pre-Processing
        self.help.append("production_companies")
        productionCompanies = self.datasetmovies['production_companies']
        productionCompanies = productionCompanies[:3799]
        productionCompanies = self.getIDFromJSON(productionCompanies, 'id', True)
        dicOfPrductioCompanies = self.handlingCatigorialVariables(productionCompanies, voteAvrg, voteCount, 1)
        productionCompanies = self.putDictToList(dicOfPrductioCompanies, productionCompanies)
        self.data.append(productionCompanies)
        '''
        # ID Pre-Processing
        self.data.append(self.basic_step_of_preprocessing('movie_id', "datasetcredits"))
        '''

        # cast Pre-Processing
        self.help.append("cast")
        cast = self.datasetcredits['cast']
        cast = cast[:3799]
        cast = self.getIDFromJSON(cast, 'id', True)
        dicOfCast = self.handlingCatigorialVariables(cast, voteAvrg, voteCount, 1)
        cast = self.putDictToList(dicOfCast, cast)

        self.data.append(cast)

        # spoken languages Pre-Processing
        self.help.append("spoken_languages")
        spokenLanguages = self.datasetmovies['spoken_languages']
        spokenLanguages = spokenLanguages[:-1]
        spokenLanguages = self.getIDFromJSON(spokenLanguages, 'iso_639_1', False)
        self.data.append(self.map_spoken_language(spokenLanguages))

        self.help.append("release_date")
        date = self.datasetmovies['release_date']
        date1 = np.copy(date)
        date = self.extractYearFromDate(date)
        self.data.append(date)

    '''
    def extractmonthFromDate(self, date):
        ans = []
        for i in range(len(date)):
            for j in range(len(date[i])):
                if date[i][j] == '/':
                        d = ""
                        j += 1
                        while date[i][j] != '/':
                            d += date[i][j]
                            j += 1
                        ans.append(int(d))
                        break
        return ans
    '''

    def extractYearFromDate(self, date):
        ans = []
        for i in range(len(date)):
            c = 0
            for j in range(len(date[i])):
                if date[i][j] == '/':
                    c += 1
                    if c == 2:
                        d = ""
                        j += 1
                        while j < len(date[i]):
                            d += date[i][j]
                            j += 1
                        ans.append(int(d))
        return ans

    def printNumOfMissing(self, columnNum):
        c = 0
        for i in range(len(self.data[columnNum])):
            if self.data[columnNum][i]:
                c += 1
        print(self.help[columnNum], 3799 - c)

        '''
                 plt.figure(figsize=(15, 8))
                 sns.distplot(self.data[5], bins=30)
                 plt.show()
        '''

    def GetData(self):
        #model = linear_model.LinearRegression()
        svr = ml.SVR(kernel='linear', C=1.0, epsilon=0.1)
        toBeTrained = []
        toBeTrained.append(self.data[1])
        toBeTrained.append(self.data[7])
        toBeTrained.append(self.data[2])
        toBeTrained.append(self.data[4])
        toBeTrained.append(self.data[6])
        toBeTrained.append(self.data[14])
        toBeTrained.append(self.data[12])
        toBeTrained = np.array(toBeTrained)
        toBeLabel = np.array(self.data[5])
        # toBeLabel = np.expand_dims(toBeLabel, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(toBeTrained.T, toBeLabel, test_size=0.30)
        # X_train = np.expand_dims(X_train, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        # X_test = np.expand_dims(X_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        # print(np.array(X_train).shape)
        # print(np.array(y_train).shape)
        svr.fit(X_train, y_train)
        #model.fit(X_train, y_train)
        prediction = svr.predict(X_test)

        # plt.scatter(X_test, y_test)
        # plt.xlabel('budged', fontsize=20)
        # plt.ylabel('vote', fontsize=20)
        # plt.plot(X_test, prediction, color='red', linewidth=3)
        print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
        # plt.show()
        return toBeTrained,toBeLabel , X_train , X_test , y_train, y_test
    def meanscale(self, colNum):

        mx = max(self.data[colNum])
        mn = min(self.data[colNum])
        meu = np.mean(self.data[colNum])
        denemurator = mx - mn
        if denemurator == 0:
            denemurator = 1

        self.data[colNum] = (self.data[colNum] - meu) / (mx - mn)

    def fillMissingData(self, columnNum):
        # Get Median
        dataCol = []
        for i in range(len(self.data[columnNum])):
            if self.data[columnNum][i]:
                dataCol.append(self.data[columnNum][i])
        dataCol = np.sort(dataCol)
        medain = 0
        if len(dataCol) % 2 == 1:
            medain = dataCol[int((len(dataCol) - 1) / 2)]
        else:
            medain = dataCol[int(len(dataCol) / 2)] + dataCol[int((len(dataCol)) / 2 - 1)]
            medain /= 2
        # fill missing data with median value
        for i in range(len(self.data[columnNum])):
            if not self.data[columnNum][i] or np.isnan(self.data[columnNum][i]):
                self.data[columnNum][i] = medain

    # calculate correlation between two column { delete row if data missied in any of them}
    def cul_correlation(self, data1, data2):
        ans1 = []
        ans2 = []
        for i in range(len(data1)):
            if data1[i] and data2[i] and (not (math.isnan(data1[i]))) and (not (math.isnan(data2[i]))):
                ans1.append(data1[i])
                ans2.append(data2[i])
        corr, _ = pearsonr(ans1, ans2)
        print('Pearsons correlation: %.3f' % corr)

    def handlingCatigorialVariables(self, arr, vote, votecount, option):
        d = dict()
        dcount = dict()
        # option =2
        if option == 1:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    if arr[i][j] in d:
                        d[arr[i][j]] += vote[i] * votecount[i]
                        dcount[arr[i][j]] += 1
                    else:
                        # d.asset(cast[i][j])
                        d[arr[i][j]] = 0
                        d[arr[i][j]] += vote[i] * votecount[i]
                        # dcount.add(cast[i][j])
                        dcount[arr[i][j]] = 1
        elif option == 2:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    if arr[i][j] in d:
                        d[arr[i][j]] += vote[i]
                        dcount[arr[i][j]] += 1
                    else:
                        # d.add(cast[i][j])
                        d[arr[i][j]] = 0
                        d[arr[i][j]] += vote[i]
                        # dcount.add(cast[i][j])
                        dcount[arr[i][j]] = 1

        for key, ele in d.items():
            count = dcount[key]
            d[key] = ele / count
        return d

    def putDictToList(self, dic, arr):
        l = []
        for i in range(len(arr)):
            x = 0
            for j in range(len(arr[i])):
                x += dic[arr[i][j]]
            l.append(x)
        return l

    def deleteMissingData(self):
        ans = [[], [], [], [], [], [], [], []]
        listOfItems = [1, 2, 4, 6, 7, 14, 5, 12]
        for i in range(len(self.data[0])):
            deleteRow = 0
            for j in range(len(listOfItems)):
                if not self.data[listOfItems[j]][i]:
                    deleteRow = 1
                    break
            if deleteRow == 0:
                # print("i am here")
                for j in range(len(listOfItems)):
                    ans[j].append(self.data[listOfItems[j]][i])
        for i in range(len(listOfItems)):
            self.data[listOfItems[i]] = ans[i]


x = PrepProcessing()
x.reformat()
print(x.help)

'''x.fillMissingData(1)
x.fillMissingData(2)
x.fillMissingData(4)
x.fillMissingData(5)
x.fillMissingData(7)
x.fillMissingData(6)'''
x.deleteMissingData()
x.meanscale(1)
x.meanscale(2)
x.meanscale(4)
x.meanscale(7)
x.meanscale(14)
x.meanscale(12)
x.meanscale(6)

x.GetData()

model = LR.Linear_Regression()
#test = SVR()
print("hello")
# correlation between vote average and month in relase date = 0.001
# correlation between vote average and years in relase date = -0.135
