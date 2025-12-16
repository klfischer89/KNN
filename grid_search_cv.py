import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age", "Glucose"]]
y = df["Outcome"]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid = {
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_leaf': [1, 3, 5, 10, 15]
}, cv = RepeatedKFold())

clf.fit(X, y)

print(clf.best_params_)
print(clf.best_score_)

clf.predict(np.array([
    [30, 25, 190]
]))