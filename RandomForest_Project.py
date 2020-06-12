
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:50:51 2020

@author: Riahi
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
dataset= pd.read_csv('C:/Users/Riahi/Desktop/PROJET_PATIE_ML/BD_Projet_NEW112- Copie.csv', delimiter=';') 
price=dataset['COST']
Data=dataset.drop(['COST'],axis=1)


x=np.array(price).reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(Data,x,test_size=0.33,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 150) 
import time
dep=time.time()
regressor.fit(X_train, Y_train)
fin=time.time()-dep

y_pred = regressor.predict(X_test)

print('R squared value', regressor.score(X_test,Y_test))


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,y_pred)
import math
print('MSE', math.sqrt(mse))

rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("RMSE: %f" % (rmse))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y_test,y_pred)
print('EV' , explained_variance_score(Y_test,y_pred))


###XGBOOST###


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
dataset= pd.read_csv('C:/Users/Riahi/Desktop/PROJET_PATIE_ML/BD_Projet_NEW112- Copie.csv', delimiter=';') 
price=dataset['COST']
Data=dataset.drop(['COST'],axis=1)

x=np.array(price).reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(Data,x,test_size=0.33,random_state=0)

import xgboost as xgb

xgb = xgb.XGBRegressor(n_estimators=10, max_depth=5, objectives = 'reg:linear' , learning_rate=0.3)
import time
dep=time.time()
xgb.fit(X_train,Y_train)
fin=time.time()-dep
predictions = xgb.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(Y_test,predictions))
print("RMSE: %f" % (rmse))
from sklearn.metrics import explained_variance_score
EV=explained_variance_score(Y_test,predictions)
print("EV : %f" %(EV))

import matplotlib.pyplot as plt
import os
os.getcwd()
os.chdir('C:/Program Files (x86)/Graphviz2.38/bin')
xgb.plot_tree(xgb,num_trees=9)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()











