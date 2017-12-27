import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier


data = pd.read_csv("../data/stats.csv", na_values="na")

# data = data.dropna()

print(data.shape)
print(data.describe())

X = data[["mu1", "sigma1", "med1", "max1", "min1", "per75_1",
          "mu2", "sigma2", "med2", "max2", "min2", "per75_2", "angle"]].as_matrix()
Y = data["label"].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# model = RandomForestClassifier(n_jobs=2, random_state=0)
# model.fit(X_train, y_train)
# classes = model.predict(X_test)
# acc = accuracy_score(y_test, classes)
#
# print(acc)
# print(model.feature_importances_)

clf = XGBClassifier()
clf.fit(X_train, y_train)
train_predictions = clf.predict_proba(X_train)[:, -1]
print(log_loss(y_train, train_predictions))

y_pred = clf.predict_proba(X_test)[:, -1]
loss = log_loss(y_test, y_pred)
print(loss)
print(clf.feature_importances_)


clf = XGBClassifier(seed=1)
clf.fit(X, Y)
pred = clf.predict_proba(X)[:, -1]
print(log_loss(Y, pred))

target_id = pd.read_json("../data/orig/train.json").as_matrix(["id"])
final = np.column_stack((target_id, pred))
csv = pd.DataFrame(final, columns=["id", "is_iceberg"])
csv.to_csv("../data/train_predicted_stats.csv", float_format='%.6f', index=False)

target = pd.read_csv("../data/test_stats.csv", na_values="na").as_matrix()
predictions = clf.predict_proba(target)[:, -1]

target_id = pd.read_json("../data/orig/test.json").as_matrix(["id"])
final = np.column_stack((target_id, predictions))

csv = pd.DataFrame(final, columns=["id", "is_iceberg"])
csv.to_csv("../data/predicted_stats.csv", float_format='%.6f', index=False)
print("Done")
