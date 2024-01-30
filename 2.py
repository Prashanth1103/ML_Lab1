import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns = housing.feature_names)
data['Price'] = housing.target

data.head()

data.isnull().sum()

data.plot()

count = data.info()

data.cov(numeric_only = True)

data.corr(numeric_only = True)

x = data.drop('Price', axis = 1)
y = data['Price']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 42)

model = SGDRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
mse, r2 = mean_squared_error(ytest, ypred), r2_score(ytest, ypred)
print(f'Mean Squared Error : {mse} \nR2 Score : {r2}')

a = 1-(mse/np.var(ytest))
print('Accuracy : ', a)

