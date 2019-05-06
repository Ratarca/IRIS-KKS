# -*- coding: utf-8 -*-
"""
    IRIS CASE

@author: Ratarca
github: /Ratarca
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

##                          </>  Check data set to understand problem   </>          ###

name_columns = df.columns
stat_table = df.describe()

#   Null values
nulls = df.isnull()
sum_nulls = nulls.sum()

#   Charts (EDA)
"""   The main charts to m.l  
    -> pairplot ; 
    -> heatmap ; 
    -> boxplot
    -> violinplot
    -> barplot
    -> jointplot
    -> residplot
    -> lineplot
    -> radviz
       """

#sns.pairplot(df,hue = 'Species',size = 3 , palette = 'Blues')       

#sns.heatmap(corr_stat,annot = True,cmap = 'Blues')

#sns.boxplot(x = 'Species' , y = 'PetalWidthCm',data=df)
       
#sns.violinplot(x = 'Species',y = 'PetalWidthCm',data=df )

#sns.barplot(x = 'Species', y = 'PetalWidthCm' , data = df)   # ax = sns.barplot(x="day", y="total_bill", hue="sex", data=tips)

#sns.jointplot(x = 'PetalWidthCm' ,y = 'SepalLengthCm',kind = 'reg',data = df )
#sns.residplot(x = 'PetalWidthCm' ,y = 'SepalLengthCm',lowess = True , data = df)



###     </> Feature engineer {Encoding + Create artificial features + normalization} </>

#Encoding
from sklearn.preprocessing import LabelEncoder
le_encoding = LabelEncoder()
df['Species'] = le_encoding.fit_transform(df['Species'])

#Create artificial features
df['area_petal'] =  df['PetalLengthCm'] * df['PetalLengthCm']
df['area_sepal'] = df['area_petal'] / df['SepalLengthCm'] 
df['flower_stranger'] = df['area_petal'] * df['area_petal']

#
X = df.drop(['Species'],axis=1)
Y = df['Species']


#Normalization
from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
sds.fit(X)
X = sds.transform(X)

#   Statistics
corr_stat = df.corr()
cov_stat = df.cov()
descr_stat = df.describe()

#sns.heatmap(corr_stat,annot=True , cmap = 'Blues')
#sns.pairplot(data = ,hue = 'Species',palette = 'inferno' )

##      </> TRAIN TEST SPLIT     </>
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4,random_state = 999)


##      </> CLASSICAL MODELS IN MACHINE LEARNING  </>
""" The main classical machine learning models are:
    ->Naive bayes
    ->KNN
    ->SVM
    ->Tree Classifier
    ->Random Forest Classifier
    ->Regression (Logistic ; Linear ; Ridge ; Elastic ; Elastic-Net)
    """
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(Xtrain,Ytrain)
Ynb_pred = nb_model.predict(Xtest)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(Xtrain,Ytrain)
Yknn_pred = knn_model.predict(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear')
svm_model.fit(Xtrain , Ytrain)
Ysvm_pred = svm_model.predict(Xtest)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_leaf_nodes = 3)
tree_model.fit(Xtrain,Ytrain)
Ytree_pred = tree_model.predict(Xtest)

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators = 300)
rfc_model.fit(Xtrain , Ytrain)
Yrfc_pred = rfc_model.predict(Xtest)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(Xtrain,Ytrain)
Ylr_pred = lr_model.predict(Xtest)

##      </> NEW CLASSICAL MODELS AND METHODs IN MACHINE LEARNING  </>
""" The main new classical machine learning models are:
    ->AdaBoost
    ->XGBOOST
    
    ->Voting Classifier (method)
    ->Stacking (method)"""
    
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(learning_rate = 0.01,n_estimators = 300)
ada_model.fit(Xtrain,Ytrain)
Yada_pred = ada_model.predict(Xtest)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.01,gamma = 2)
xgb_model.fit(Xtrain,Ytrain)
Yxgb_pred = xgb_model.predict(Xtest)

from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(learning_rate = 0.01)
lgbm_model.fit(Xtrain,Ytrain)
Ylgbm_pred = nb_model.predict(Xtest)

from sklearn.ensemble import VotingClassifier
vtg_model = VotingClassifier(estimators = [ ('nb',nb_model),('knn',knn_model),('svm',svm_model),('rfc',rfc_model),('tree',tree_model)
                                           ,('lr',lr_model),('ada',ada_model),('lgbm',lgbm_model),('xgb',xgb_model)] , voting = 'hard')
    
    
## </> COMPARSION AND TEST's
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold_use = KFold(n_splits = 5 , random_state = 999)

std_metric =[]
mean_metric =[]
accuracy =[]

models_name = ['nb_model', 'knn_model' ,'svm_model', 'rfc_model', 'tree_model', 'lr_model' ,'ada_model', 'lgbm_model','xgb_model', 'vtg_model']
models_work = [nb_model,knn_model,svm_model,rfc_model,tree_model,lr_model,ada_model,lgbm_model,xgb_model,vtg_model]

for models in models_work:
    models = models 
    cv_result = cross_val_score(models , X,Y,cv = kfold_use , scoring = 'accuracy')
    cv_result = cv_result

    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)

box_model = {'mean':mean_metric,'std':std_metric,'name_model':models_name}
sns.boxplot(x= models_name,y=accuracy,palette ='inferno')


box_model_df = pd.DataFrame(box_model)

##      </> DEEP LEARNING MODELS  </>
""" The main new classical machine learning models are:
    NEXT TRAIN
    """


