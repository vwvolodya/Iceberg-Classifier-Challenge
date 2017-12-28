import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier


data = pd.read_csv("../data/stats.csv", na_values="na")

print(data.shape)
print(data.describe())

X = data[["mu1", "sigma1", "med1", "max1", "min1", "per75_1",
          "mu2", "sigma2", "med2", "max2", "min2", "per75_2", "angle"]].as_matrix()
Y = data["label"].as_matrix()

skf = StratifiedKFold(n_splits=4, random_state=41)
models = []
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    train_predictions = clf.predict_proba(X_train)[:, -1]
    print(log_loss(y_train, train_predictions))
    y_pred = clf.predict_proba(X_test)[:, -1]
    y_pred_class = clf.predict(X_test)
    loss = log_loss(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_class)
    print(loss, acc)
    print(clf.feature_importances_)

    models.append(clf)


def infer(mdls, dt):
    pred = [c.predict_proba(dt)[:, -1] for c in mdls]
    pred = np.column_stack(pred)
    pred = np.mean(pred, axis=1)
    return pred


def get_data_frame(predictions, is_test=False):
    if is_test:
        target_id = pd.read_json("../data/orig/test.json").as_matrix(["id"])
    else:
        target_id = pd.read_json("../data/orig/train.json").as_matrix(["id"])
    final = np.column_stack((target_id, predictions))
    csv = pd.DataFrame(final, columns=["id", "is_iceberg"])
    return csv


pred = infer(models, X)
print(log_loss(Y, pred))
csv = get_data_frame(pred, is_test=False)
csv.to_csv("../data/train_predicted_stats.csv", float_format='%.6f', index=False)

target = pd.read_csv("../data/test_stats.csv", na_values="na").as_matrix()
test_predictions = infer(models, target)
csv = get_data_frame(test_predictions, is_test=True)
csv.to_csv("../data/predicted_stats.csv", float_format='%.6f', index=False)
print("Done")
