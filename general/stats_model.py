import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


data = pd.read_csv("../data/stats.csv", na_values="na")

data = data.dropna()

print(data.shape)
print(data.describe())

X = data[["mu1", "sigma1", "med1", "max1", "min1", "per75_1",
          "mu2", "sigma2", "med2", "max2", "min2", "per75_2", "angle"]].as_matrix()
Y = data["label"].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# model = RandomForestClassifier(n_jobs=2, random_state=0)
# model.fit(X_train, y_train)
# classes = model.predict(X_test)
# acc = accuracy_score(y_test, classes)
#
# print(acc)
# print(model.feature_importances_)

clf = XGBClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_1 = accuracy_score(y_test, y_pred)
print(acc_1)
print(clf.feature_importances_)
