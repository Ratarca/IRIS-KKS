# -*- coding: utf-8 -*-
"""
    IRIS CASE 00_TRAIN

@author: Ratarca
github: CavalcanteRafael
linkedin :rafael-cavalcante-4b58b2100
"""

path = "C:/Users/Rafael/Desktop/IRIS/Source_Data/"
file = "Iris.csv"

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


df_import = pd.read_csv(path + file)

df = df_import.drop(["Id"],axis = 1)

#Check data set to understand problem

name_columns = df.columns
stat_table = df.describe()

nulls = df.isnull()
sum_nulls = nulls.sum()

corr_table = df.corr()
cov_table = df.cov()

list_begginer_bescribe = [name_columns , stat_table , sum_nulls]


#To print basic things
for i in list_begginer_bescribe:
    print(i,"\n")

#detect outliers

###@@@@###EDA : <pairplot ; violinplot ; jointplot ; catplot ; histogram ;scatterplot ; barplot ; heatmap
    
    #sns.pairplot(data = df , hue = columns ,palette = 'inferno') or sns.pairplot(data = df , hue = columns ,vars = [columns*] , palette = 'inferno') 
#sns.pairplot(data = df , hue ='Species' ,palette = "inferno")
    #sns.violinplot(data = df ,x = , y = , hue = ,palette = 'inferno')
#sns.violinplot(data = df ,x = 'Species', y = 'PetalLengthCm',palette = 'inferno')
    #sns.jointplot
    

###@@@@### Pivot Analysis



###@@@@###Encoding & Feature Engineer
from sklearn.preprocessing import LabelEncoder
le_encoding = LabelEncoder()
df['Species'] = le_encoding.fit_transform(df['Species'])


#Artificial features
df['area_petal'] = df['PetalLengthCm'] * df['PetalLengthCm']
df['area_sepal'] = df['area_petal'] / df['SepalLengthCm'] 
df['flower_stranger'] = df['area_petal'] * df['area_petal']


corr_table = df.corr()
cov_table = df.cov()

X = df.drop(['Species'],axis=1)
Y = df['Species']

from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
sds.fit(X)
X = sds.transform(X)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

###@@@@###Classical algoritms
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(Xtrain,Ytrain)
Ynb_pred = nb_model.predict(Xtest)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 2)
knn_model.fit(Xtrain,Ytrain)
Yknn_pred = knn_model.predict(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear')
svm_model.fit(Xtrain,Ytrain)
Ysvm_pred = svm_model.predict(Xtest)

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators = 300)
rfc_model.fit(Xtrain,Ytrain)
Yrfc_pred = rfc_model.predict(Xtest)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_leaf_nodes = 3)
tree_model.fit(Xtrain,Ytrain)
Ytree_pred = tree_model.predict(Xtest)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(Xtrain,Ytrain)
Ylr_pred = lr_model.predict(Xtest)
    
###@@@@### News algoritms
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(learning_rate = 0.01,n_estimators = 300)
ada_model.fit(Xtrain,Ytrain)
Yada_pred = ada_model.predict(Xtest)


from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(learning_rate = 0.01)
lgbm_model.fit(Xtrain,Ytrain)
Ylgbm_pred = nb_model.predict(Xtest)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.01,gamma = 2)
xgb_model.fit(Xtrain,Ytrain)
Yxgb_pred = xgb_model.predict(Xtest)

from sklearn.ensemble import VotingClassifier
vtg_model = VotingClassifier(estimators = [ ('nb',nb_model),('knn',knn_model),('svm',svm_model),('rfc',rfc_model),('tree',tree_model)
                                           ,('lr',lr_model),('ada',ada_model),('lgbm',lgbm_model),('xgb',xgb_model)] , voting = 'hard')


###@@@@###Cross validation & Comparsion
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold_use = KFold(n_splits = 5,random_state = 999)

std_metric = []
mean_metric = []
accuracy = []

models_name = ['nb_model', 'knn_model' ,'svm_model', 'rfc_model', 'tree_model', 'lr_model' ,'ada_model', 'lgbm_model','xgb_model', 'vtg_model']
models_work = [nb_model,knn_model,svm_model,rfc_model,tree_model,lr_model,ada_model,lgbm_model,xgb_model,vtg_model]

for model in models_work:
    model = model
    cv_result = cross_val_score(model ,X,Y,cv = kfold_use,scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'mean':mean_metric ,'std': std_metric,'name': models_name})
sns.boxplot(x = models_name , y = accuracy , palette = 'inferno')
