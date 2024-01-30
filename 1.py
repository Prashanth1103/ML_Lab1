import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns = housing.feature_names)

data['Price'] = housing.target

print("First 5 rows : \n")
data.head()

print("Checking for null values : ")
data.isnull().sum()

data.plot()
plt.show()

print("Covariance Matrix : ")
data.cov()

print("Correlation Matrix : ")
data.corr()

x = data.drop('Price', axis = 1)
y = data['Price']
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.25, random_state = 42)

model = LinearRegression()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print(f'\n Mean Squared Error : {mse}')
r2 = r2_score(ytest, ypred)
print(f'\n R2 Score : {r2}')

a = 1-(mse/np.var(ytest))
print(a)
