import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score


X = pd.read_json("../data/orig/train.json")
Y = X.as_matrix(["is_iceberg"])
band1 = X.as_matrix(["band_1"])
band2 = X.as_matrix(["band_2"])
res = []
for i in range(len(band1)):
    current1 = band1[i][0]
    current2 = band2[i][0]
    r = np.array([current1, current2])
    row = np.mean(r, axis=0).tolist()
    res.append(row)
X = np.array(res)


pca = PCA(n_components=120, random_state=78)
X = pca.fit_transform(X)

skf = KFold(n_splits=4, random_state=92)
models = []
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    mdl = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=52)
    mdl.fit(X_train, y_train)
    probs = mdl.predict_proba(X_train)[:, -1]
    classes = mdl.predict(X_train)
    test_classes = mdl.predict(X_test)
    test_probs = mdl.predict_proba(X_test)[:, -1]
    loss = log_loss(y_train, probs)
    test_loss = log_loss(y_test, test_probs)
    acc = accuracy_score(y_train, classes)
    test_acc = accuracy_score(y_test, test_classes)
    print(loss, test_loss, acc, test_acc)


