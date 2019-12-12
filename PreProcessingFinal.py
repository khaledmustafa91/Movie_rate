import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt


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
        for i in range (len(arr)):
            aset.add(arr[i])
        counter =1
        for setElement in aset:
            dic[setElement]=counter
            counter+=1
        return dic


    def mapping(self, dic, arr):
        for i in range(len(arr)):
            arr[i] = dic[arr[i]]
        return arr


    def getIDFromJSON(self, arr,key,int_str):
        for i in range(len(arr)):
            str1 = arr[i].replace(']', '').replace('[', '')
            l = str1.replace('"', '').replace(',', '').replace('{', '').replace('}', '').replace(':', '')
            l = l.split(' ')
            listOfOneElement = []
            if key == 'Directing' or key == 'production':
                for j in range(len(l)):
                    if(l[j] == 'id'):
                        id = int(l[j + 1])
                        j += 1
                        while j < len(l):
                            if l[j] == key:
                                listOfOneElement.append(id)
                                break
                            if l[j] == "id":
                                id = int(l[j+1])
                            j += 1
            else:
                for j in range(len(l)):
                    if l[j] == key :
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
        self.data.append(self.basic_step_of_preprocessing('budget',"datasetmovies"))

        self.help.append("revenue")
        self.data.append(self.basic_step_of_preprocessing('revenue', "datasetmovies"))

        self.help.append("original_language")
        org_lang = self.basic_step_of_preprocessing('original_language',"datasetmovies")
        self.dicOfLanguages = self.transform_map(org_lang)
        org_lang = self.mapping(self.dicOfLanguages, org_lang)
        tmp = []
        for i in range(len(org_lang)):
            tmp.append(int(org_lang[i]))
        self.data.append(tmp)

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
        genres =self.getIDFromJSON(genres[:], 'id', True)
        dicOfgenres = self.handlingCatigorialVariables(genres, voteAvrg, voteCount, 2)
        genres = self.putDictToList(dicOfgenres, genres)
        self.data.append(genres)

        self.help.append("keywords")
        keywords = self.datasetmovies['keywords']
        keywords = keywords[:3799]
        keywords = self.getIDFromJSON(keywords,'id',True)
        dicOfkeywords = self.handlingCatigorialVariables(keywords, voteAvrg, voteCount, 2)
        keywords = self.putDictToList(dicOfkeywords, keywords)
        self.data.append(keywords)

        # Production Countries Pre-Processing
        self.help.append("production_countries")
        productionCountries = self.datasetmovies['production_countries']
        productionCountries = productionCountries[:3799]
        productionCountries = self.getIDFromJSON(productionCountries, "iso_3166_1", False)
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
        dicOfCast = self.handlingCatigorialVariables(cast,voteAvrg,voteCount,1)
        cast = self.putDictToList (dicOfCast,cast)
        self.data.append(cast)

        # spoken languages Pre-Processing
        self.help.append("spoken_languages")
        spokenLanguages = self.datasetmovies['spoken_languages']
        tmp = np.copy(spokenLanguages)
        spokenLanguages = self.getIDFromJSON(tmp, 'iso_639_1', False)
        spokenLanguages = self.map_spoken_language(spokenLanguages)
        dicOfspokenLanguages = self.handlingCatigorialVariables(spokenLanguages, voteAvrg, voteCount, 2)
        spokenLanguages = self.putDictToList(dicOfspokenLanguages, spokenLanguages)
        self.data.append(spokenLanguages)


        self.help.append("release_date")
        date = self.datasetmovies['release_date']
        date1 = np.copy(date)
        date = self.extractYearFromDate(date)
        date1 = self.extractmonthFromDate(date1)
        self.data.append(date)

        self.help.append("homePage")
        homePages = self.datasetmovies['homepage']
        dicOfHomePage = dict()
        mappedHomePages = []
        c = 0
        for i in range(len(homePages)):
            if homePages[i] in dicOfHomePage:
                mappedHomePages.append(dicOfHomePage[homePages[i]])
            else:
                c += 1
                dicOfHomePage[homePages[i]] = c
                mappedHomePages.append(c)
        self.data.append(mappedHomePages)

        self.help.append("month")
        self.data.append(date1)


        crew = self.datasetcredits["crew"]
        director = np.copy(crew)
        production = np.copy(crew)

        # read crew ; delete missing ; selet id ; count them
        self.help.append("crew")
        tmp = []
        for i in range(len(crew)):
            if crew[i]:
                tmp.append(crew[i])
        crew1 = self.getIDFromJSON(tmp, 'id', True)
        tmp = []
        for i in crew1:
            tmp.append(len(i))
        self.data.append(tmp)

        self.help.append("Director")
        tmp = np.copy(director)
        director = self.getIDFromJSON(tmp, 'Directing', False)
        dicOfdirector = self.handlingCatigorialVariables(director, voteAvrg, voteCount, 2)
        director = self.putDictToList(dicOfdirector, director)
        self.data.append(director)

        self.help.append("production")
        tmp = np.copy(production)
        production = self.getIDFromJSON(tmp, 'Production', False)
        dicOfproduction = self.handlingCatigorialVariables(production, voteAvrg, voteCount, 2)
        production = self.putDictToList(dicOfproduction, production)
        self.data.append(production)



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


    def printNumOfMissing(self,columnNum):
        c = 0
        for i in range(len(self.data[columnNum])):
            if self.data[columnNum][i]:
                c += 1
        print(self.help[columnNum], 3799-c)


    def GetData(self):
        model = linear_model.LinearRegression()
        toBeTrained = []
        #toBeTrained.append(self.data[0]) # --> id
        #toBeTrained.append(self.data[1]) # --> budget

        toBeTrained.append(self.data[2]) # -->revenue

        #toBeTrained.append(self.data[3]) # -->original_language


        toBeTrained.append(self.data[4]) # -->vote_count
        toBeTrained.append(self.data[6]) # -->runtime
        toBeTrained.append(self.data[7]) # -->popularity
        toBeTrained.append(self.data[8]) # -->genres
        toBeTrained.append(self.data[9]) # -->keywords

        #toBeTrained.append(self.data[10]) # -->production_countries

        toBeTrained.append(self.data[11]) # -->production_companies
        toBeTrained.append(self.data[12]) # -->cast

        #toBeTrained.append(self.data[13]) # -->spokenLanguage
        #toBeTrained.append(self.data[14]) # -->release_date
        #toBeTrained.append(self.data[15]) # --> homePage
        #toBeTrained.append(self.data[17]) # --> crew number

        toBeTrained.append(self.data[18]) # --> Director

        #toBeTrained.append(self.data[19]) # --> production
        #toBeTrained.append(self.data[16]) # --> month in date

        toBeTrained = np.array(toBeTrained)
        toBeLabel = np.array(self.data[5])
        X_train, X_test, y_train, y_test = train_test_split(toBeTrained.T, toBeLabel, test_size=0.30)
        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
#        print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

        plt.figure("Test")
        plt.scatter(np.atleast_2d(X_test[:,0]), y_test)

     #   plt.scatter(np.atleast_2d(X_train[:,0]), y_train)
        plt.plot(np.atleast_2d(X_test[:,0]), prediction, color='red', linewidth=3)
        plt.show()

        return toBeTrained,toBeLabel


    def meanNormalization(self):
        for i in range(len(self.data)):
            if i != 5:
                mx = max(self.data[i])
                mn = min(self.data[i])
                meu = np.mean(self.data[i])
                self.data[i] = (self.data[i] - meu) / (mx - mn)


    def fillMissingData(self):
        # Get Median
        for i in range(len(self.data)):
            if i != 5:
                dataCol = []
                for j in range(len(self.data[i])):
                    if self.data[i][j]:
                        dataCol.append(self.data[i][j])
                dataCol = np.sort(dataCol)
                medain = 0
                if len(dataCol)%2 == 1:
                    medain = dataCol[int((len(dataCol) - 1)/2)]
                else:
                    medain = dataCol[int(len(dataCol)/2)] + dataCol[int((len(dataCol))/2 -1)]
                    medain /= 2
                # fill missing data with median value
                for j in range(len(self.data[i])):
                    if not self.data[i][j] or np.isnan(self.data[i][j]):
                        self.data[i][j] = medain


    # calculate correlation between two column { delete row if data missied in any of them}
    def cul_correlation(self, data1, data2):
        ans1 = []
        ans2 = []
        for i in range(len(data2)):
            if data1[i] and data2[i] and (not(math.isnan(data1[i])) )and (not (math.isnan(data2[i]))) :
                ans1.append(data1[i])
                ans2.append(data2[i])
        corr, _ = pearsonr(ans1,ans2)
        print('Pearsons correlation: %.3f' % corr)



    def handlingCatigorialVariables(self,arr,vote,votecount,option):
        d = dict()
        dcount = dict()
        option = 2
        if option == 1:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    if arr[i][j] in d:
                        d[arr[i][j]] +=vote[i]*votecount[i]
                        dcount[arr[i][j]] += 1
                    else :
                        d[arr[i][j]] = 0
                        d[arr[i][j]] += vote[i]*votecount[i]
                        dcount[arr[i][j]] = 1
        elif option == 2:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    if arr[i][j] in d:
                        d[arr[i][j]] += vote[i]
                        dcount[arr[i][j]] += 1
                    else:
                        d[arr[i][j]] = 0
                        d[arr[i][j]] += vote[i]
                        dcount[arr[i][j]] = 1

        for key,ele in d.items():
            count = dcount[key]
            d[key] = ele / count
        return d

    def putDictToList (self,dic,arr):
        l = []
        for i in range(len(arr)):
            x = []
            for j in range(len(arr[i])):
                x.append(dic[arr[i][j]])
            x = np.sort(x)
            tmp = 0
            if len(arr[i]) != 0:
                if len(x) % 2 == 1:
                    tmp = x[(len(x) - 1) // 2]
                else:
                    tmp = x[len(x) // 2] + x[len(x) // 2 - 1]
                    tmp /= 2
                l.append(tmp)
            else:
                l.append(0)
        return l

    def deleteMissingData(self):
        ans = [[], [], [], [], [], [], [], [] , [], []]
        listOfItems = [2, 4,5,  6, 7, 8, 9, 11, 12, 18]
        #ans = [[], [], [], [], [], [], [] , [], [], [], [], [], [], [], [], [], [], [], [], []]
        #listOfItems = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,  14, 15, 16, 17, 18, 19]
        for i in range(len(self.data[0])):
            deleteRow = 0
            for j in range(len(listOfItems)):
                if not self.data[listOfItems[j]][i]:
                    deleteRow = 1
                    break
            if deleteRow == 0:
                for j in range(len(listOfItems)):
                    ans[j].append(self.data[listOfItems[j]][i])
        for i in range(len(listOfItems)):
            self.data[listOfItems[i]] = ans[i]


x = PrepProcessing()
x.reformat()
print(x.help)
#x.fillMissingData()   #fill missing data with median
x.deleteMissingData()
x.meanNormalization()
x.cul_correlation(x.data[0], x.data[5])
x.cul_correlation(x.data[1], x.data[5])
x.cul_correlation(x.data[2], x.data[5])
x.cul_correlation(x.data[3], x.data[5])
x.cul_correlation(x.data[4], x.data[5])
x.cul_correlation(x.data[6], x.data[5])
x.cul_correlation(x.data[7], x.data[5])
x.cul_correlation(x.data[8], x.data[5])
x.cul_correlation(x.data[9], x.data[5])
x.cul_correlation(x.data[10], x.data[5])
x.cul_correlation(x.data[11], x.data[5])
x.cul_correlation(x.data[12], x.data[5])
x.cul_correlation(x.data[13], x.data[5])
x.cul_correlation(x.data[14], x.data[5])
x.cul_correlation(x.data[15], x.data[5])
x.cul_correlation(x.data[16], x.data[5])
x.cul_correlation(x.data[17], x.data[5])
x.cul_correlation(x.data[18], x.data[5])
x.cul_correlation(x.data[19], x.data[5])

x.GetData()
# correlation between vote average and month in relase date = 0.001
# correlation between vote average and years in relase date = -0.135
