from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.DataFrame(data = pd.read_csv(r'C:\users\prash\downloads\iris.csv'))

print(data.head())

X=data.drop(["species"],axis=1)
y=data["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)

model=KNeighborsClassifier()
model.fit(X,y)
y_pred=model.predict(X_test)
print(y_pred)
a=accuracy_score(y_test,y_pred)
print(f"the accuracy is : {a}")

