import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = DecisionTreeClassifier(max_depth = 2)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

plot_tree(model, 
          feature_names = ["BMI", "Age"], 
          class_names = ["Kein Diabetes", "Diabetes"],
          filled = True)
plt.show()