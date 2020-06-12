# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 02:57:23 2020

@author: Riahi
"""

import pandas as pd
import numpy as np
dataset= pd.read_csv('C:/Users/Riahi/Desktop/PROJET_PATIE_ML/BD_Projet_NEW112- Copie.csv', delimiter=';') 
price=dataset['COST']
Real_regression=dataset['ID_PRODUCT_TYPE']

x=np.array(price).reshape(-1,1)
y=np.array(Real_regression).reshape(-1,1)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

print(model.coef_[0]) ###la valeur de a
print(model.intercept_[0]) ### la valeur de b

y_pred=model.predict(x)
import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue') ### les points
plt.plot(x, model.predict(x), color='red') ### droite de prediction
plt.title('Linear Regression')
plt.xlabel('price')
plt.ylabel('Real_regression')


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y, y_pred)
import math
print('RMSE', math.sqrt(mse))
print("R squared value", model.score(x,y))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(x,y)
print('EV' , explained_variance_score(x,y))
import time
dep=time.time()
fin=time.time()-dep
 

from sklearn.metrics import r2_score
r2=r2_score(y, y_pred)

print('prediction for test step' ,model.predict([[5000]]))




 ## Reg multi##
 
 
 
 
price=dataset['COST']
Data=dataset.drop(['COST'], axis=1)
Y=np.array(price).reshape(-1,1)



from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(Data,Y)

print(model.coef_[0]) ###la valeur de a
print(model.intercept_[0]) ### la valeur de b

X=np.array(Data)
print("R squared value", model.score(Data,Y))
ypred=model.predict(Data)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y, ypred)
print('mse',mean_squared_error(Y, ypred) )
import math
print('RMSE', math.sqrt(mse))
 
from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Data,Y)
print('EV' , explained_variance_score(Data,Y))
import time
dep=time.time()
fin=time.time()-dep

from sklearn.metrics import r2_score
r2=r2_score(y, y_pred)
print('r2',r2_score(y, y_pred) )



