#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:16:07 2018

@author: Arunkumar
"""

import numpy as np
import pandas as pd

df = pd.read_csv("Heart.csv")
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

x = np.delete(x,87,0)
x = np.delete(x,265,0)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
y=le.fit_transform(y)
x[:,-1]=le.fit_transform(x[:,-1])

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN',strategy = 'most_frequent',axis=0)
x[:,11:12]=imputer.fit_transform(x[:,11:12])

onehotencoder = OneHotEncoder(categorical_features = [2,12])
x = onehotencoder.fit_transform(x).toarray()
x = np.delete(x,0,1)
x = np.delete(x,3,1)

y=y[2:]

import statsmodels.formula.api as sm
#x=np.append(arr = np.ones((301,1)).astype(int), values = x,axis=1)
#x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
#x_modeled = backwardElimination(x_opt,0.7)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0 )

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print('FOR DECISION TREE CLF :',accuracy_score(y_test,y_pred))


'''def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x'''

from sklearn.svm import SVC
sclf=SVC(kernel='poly',gamma='auto')
sclf.fit(x_train,y_train)
y_pred2 = sclf.predict(x_test)
print('FOR SVC :',accuracy_score(y_test,y_pred2))
.

 
