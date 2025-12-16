import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age", "Glucose"]]
y = df["Outcome"]

kf = RepeatedKFold(n_splits = 5, n_repeats = 100)

train_scores = []
test_scores = []

for train_index, test_index in kf.split(X):
    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    X_test  = X.loc[test_index]
    y_test  = y.loc[test_index]
    
    model = DecisionTreeClassifier(max_depth = 4)
    model.fit(X_train, y_train)

    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
    
print(np.mean(train_scores))
print(np.mean(test_scores))