import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age", "Glucose"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

clf = GridSearchCV(DecisionTreeClassifier(), param_grid = {
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_leaf': [1, 3, 5, 10, 15]
}, cv = RepeatedKFold())

clf.fit(X_train, y_train)

print(clf.best_params_)
print(clf.best_score_)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))