#-*-coding:utf8;-*-
#qpy:3
#qpy:consol
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data

dataset=pd.read_csv('filename')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#split the data into train and test data 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


#impliment the classifier

from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression
simplelinearRegression.fit(X_train,y_train)

#you can put the value of X_test to find the prediction
y_predict=simplelinearRegression.predict(X_test)
plt.scatter(X_train,y_train='red')

plt.plot(X_train,simplelinearRegrssion.predict(X_train))
plt.show()
