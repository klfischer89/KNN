import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = KNeighborsClassifier(n_neighbors = 29, p = 1)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

bmi_min = df["BMI"].min()
bmi_max = df["BMI"].max()

age_min = df["Age"].min()
age_max = df["Age"].max()

bmi_range = np.arange(bmi_min, bmi_max, 0.1)
age_range = np.arange(age_min, age_max, 0.1)

xx, yy = np.meshgrid(bmi_range, age_range)

X_pred = np.c_[xx.ravel(), yy.ravel()]
zz = model.predict_proba(X_pred)[:, 1]

zz = zz.reshape(xx.shape)

plt.contourf(xx, yy, zz, alpha = 0.5)
plt.scatter(X_train["BMI"], X_train["Age"], c = y_train, s = 5)

plt.contourf(xx, yy, zz, alpha = 0.5)
plt.scatter(X_test["BMI"], X_test["Age"], c = y_test, s = 5)
plt.show()