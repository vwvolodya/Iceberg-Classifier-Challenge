import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def make_x(path1, path2):
    data_inception = pd.read_csv(path1)
    data_lenet = pd.read_csv(path2)
    dt = data_inception.join(data_lenet["is_iceberg"], rsuffix="_2")
    res = dt.as_matrix(["is_iceberg", "is_iceberg_2"])
    return res


def evaluate(mdl, data, labels):
    probs = mdl.predict_proba(data)
    loss = log_loss(labels, probs[:, 1])
    return loss


p1 = "../data/train_predicted_inception.csv"
p2 = "../data/train_predicted_lenet.csv"
x = make_x(p1, p2)
data = pd.read_json("../data/orig/train.json")
y = data["is_iceberg"].as_matrix()


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = LogisticRegression(random_state=124, verbose=1, max_iter=150)
model.fit(X_train, y_train)

print(evaluate(model, X_train, y_train))
print(evaluate(model, X_test, y_test))


model = LogisticRegression(random_state=124, verbose=1, max_iter=150)
model.fit(x, y)

print(evaluate(model, x, y))

target = make_x("../data/predicted_inception.csv", "../data/predicted_lenet.csv")
target_id = pd.read_json("../data/orig/test.json").as_matrix(["id"])
final_predictions = model.predict_proba(target)
final_predictions = final_predictions[:, -1]

final = np.column_stack((target_id, final_predictions))
csv = pd.DataFrame(final, columns=["id", "is_iceberg"])
csv.to_csv("../data/stacked.csv", float_format='%.6f', index=False)
print("Done")
