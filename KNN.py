import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = KNeighborsClassifier(n_neighbors = 15, p = 1)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(model.predict_proba(X_test))