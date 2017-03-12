# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:48:58 2017

@author: Saurabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

df = pd.read_csv("ex1data1.csv")
df.columns = ['X','Y']
print(df.head())
print()
print()
X = np.array(df.drop(['Y'],1))
Y = np.array(df.Y)

#Feature Scaling
Scaled_X = preprocessing.scale(X)
Scaled_Y = preprocessing.scale(Y)


#Taking 75% of Scaled_X and Scaled_Y for training and rest 25% for testing purpose
X_train , X_test ,  Y_train , Y_test = cross_validation.train_test_split(Scaled_X,Scaled_Y,test_size=0.25)

#Shape of X_train should be array(number_of_sample,number_of_feature) 
#If you have only one feature and shape of array is not like (number_of_sample,1) , then reshape --> X_train.reshape(len(X_train),1)

#Now here comes linear regression
clf = LinearRegression()
clf.fit(X_train,Y_train)  #Training

plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,clf.predict(X_train),color="green")

accuracy = clf.score(X_test,Y_test)
print("Accuracy:",accuracy*100,"%")

