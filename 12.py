import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\users\prash\Downloads\iris.csv")
data = pd.DataFrame(df)

data.head()

data.isnull().sum()

x = data.drop('species', axis = 1)
y = data['species']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = SVC(kernel = 'linear', C = 1.0, gamma = 'auto')
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)

accuracy = accuracy_score(ytest, ypred)
confusion_mat = confusion_matrix(ytest, ypred)
classification_rep = classification_report(ytest, ypred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)

