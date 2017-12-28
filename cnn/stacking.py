import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score


def make_x(*args):
    frames = [pd.read_csv(p)["is_iceberg"] for p in args]
    dt = pd.DataFrame(frames).transpose()
    res = dt.as_matrix()
    return res


def evaluate(mdl, data, labels):
    probs = mdl.predict_proba(data)
    classes = mdl.predict(data)
    probs = probs[:, -1]
    loss = log_loss(labels, probs)
    acc = accuracy_score(labels, classes)
    return loss, acc


p1 = "../data/train_predicted_inception.csv"
p2 = "../data/train_predicted_lenet.csv"
p3 = "../data/train_predicted_stats.csv"
x = make_x(p1, p2, p3)
data = pd.read_json("../data/orig/train.json")
y = data["is_iceberg"].as_matrix()


skf = StratifiedKFold(n_splits=4, random_state=81)
models = []
for train, test in skf.split(x, y):
    X_train, X_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    # model = LogisticRegression(random_state=24, verbose=0, max_iter=150)
    model = SVC(random_state=20, probability=True)
    model.fit(X_train, y_train)
    print("Fold #")
    print(evaluate(model, X_train, y_train))
    print(evaluate(model, X_test, y_test))
    models.append(model)


def get_avg_(mdls, dt):
    final_predictions = [m.predict_proba(dt)[:, -1] for m in mdls]
    res = np.column_stack(final_predictions)
    res = np.mean(res, axis=1)
    return res


target = make_x("../data/predicted_inception.csv", "../data/predicted_lenet.csv", "../data/predicted_stats.csv")
target_id = pd.read_json("../data/orig/test.json").as_matrix(["id"])

final = get_avg_(models, target)
final = np.column_stack((target_id, final))
csv = pd.DataFrame(final, columns=["id", "is_iceberg"])
csv.to_csv("../data/stacked.csv", float_format='%.6f', index=False)
print("Done")
