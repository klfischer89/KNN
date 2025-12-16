import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age", "Glucose", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

sc = StandardScaler()
sc.fit(X_train)

clf = GridSearchCV(KNeighborsClassifier(), param_grid = {
    'n_neighbors': [3, 5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175]
}, cv = RepeatedKFold())

clf.fit(sc.transform(X_train), y_train)

print(clf.score(sc.transform(X_train), y_train))
print(clf.score(sc.transform(X_test), y_test))