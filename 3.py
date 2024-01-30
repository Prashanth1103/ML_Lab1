import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\users\prash\Downloads\iris.csv")

df.head()

df.isnull().sum()

df.cov(numeric_only=True)

df.corr(numeric_only=True)

x=df.drop('species',axis=1)
y=df['species']
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 1, random_state = 42)

model = LogisticRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
a = accuracy_score(ytest, ypred)
print("Accuracy : ", a)

