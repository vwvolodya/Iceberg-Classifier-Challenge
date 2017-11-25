import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def split(path):
    data = pd.read_json(path)
    np_data = data[["band_1", "band_2", "inc_angle", "is_iceberg"]].as_matrix()
    y = data["is_iceberg"].as_matrix()
    stk = StratifiedKFold(n_splits=4, random_state=101)
    result = []
    for train, test in stk.split(np_data, y):
        fold_train = np_data[train, :]
        fold_test = np_data[test, :]
        print(fold_train.shape)
        print(fold_test.shape)
        result.append((fold_train, fold_test))
    return result


def scale_data(path):
    data = pd.read_json(path)
    np_data = data[["band_1", "band_2"]].as_matrix()
    min_1 = float("inf")
    min_2 = float("inf")
    max_1 = float("-inf")
    max_2 = float("-inf")
    for i in range(np_data.shape[0]):
        el_1 = np_data[i, 0]
        el_2 = np_data[i, 1]
        current_min_1 = min(el_1)
        current_min_2 = min(el_2)

        current_max_1 = max(el_1)
        current_max_2 = max(el_2)

        min_1 = min(min_1, current_min_1)
        min_2 = min(min_2, current_min_2)

        max_1 = max(max_1, current_max_1)
        max_2 = max(max_2, current_max_2)
    print("Min 1 ", min_1)
    print("Min 2 ", min_2)
    print("Max 1", max_1)
    print("Max 2", max_2)
    return np_data


if __name__ == "__main__":
    original = "../data/orig/train.json"
    # res = split(original)
    # for i, (tr, tt) in enumerate(res):
    #     np.save("../data/folds/train_%s" % i, tr)
    #     np.save("../data/folds/test_%s" % i, tt)
    scale_data(original)
    print("Finished!")
