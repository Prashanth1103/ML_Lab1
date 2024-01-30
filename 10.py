import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data=pd.read_csv(r"C:\users\prash\downloads\housing.csv")
print(data.head())
count=data.count()
print(count)

print(list(data.columns))

X=data.drop(['24.00'],axis=1)
y=data["24.00"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.09,random_state = 42)
model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
a = 1 - (mse / np.var(y_test))
print(f"the accuracy of the model is : {a}")
plt.plot(y_test[1:10],y_pred[1:10])
plt.xlabel("actual value")
plt.ylabel("predicted value")
plt.show()

