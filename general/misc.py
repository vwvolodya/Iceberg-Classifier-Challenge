import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def simple_split(path):
    data = np.load(path)
    train, test = train_test_split(data, test_size=0.2, random_state=101)
    np.save("../data/for_train_encoder", train)
    np.save("../data/for_test_encoder", test)


def split(path, n_splits=5):
    data = pd.read_json(path)
    np_data = data[["band_1", "band_2", "inc_angle", "is_iceberg"]].as_matrix()
    y = data["is_iceberg"].as_matrix()
    stk = StratifiedKFold(n_splits=n_splits, random_state=42)
    result = []
    for train, test in stk.split(np_data, y):
        fold_train = np_data[train, :]
        fold_test = np_data[test, :]
        print(fold_train.shape)
        print(fold_test.shape)
        result.append((fold_train, fold_test))
    return result


def get_min_max(path):
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


def get_mu_sigma(path):
    data = pd.read_json(path)
    np_data = data[["band_1", "band_2"]].as_matrix()
    band_1 = []
    band_2 = []
    for i in range(np_data.shape[0]):
        el_1 = np_data[i, 0]
        el_2 = np_data[i, 1]
        band_1.append(el_1)
        band_2.append(el_2)

    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    mu_1 = np.mean(band_1)
    mu_2 = np.mean(band_2)
    sigma_1 = np.std(band_1)
    sigma_2 = np.std(band_2)
    print("Mu 1 ", mu_1)
    print("Mu 2 ", mu_2)
    print("Sigma 1", sigma_1)
    print("Sigma 2", sigma_2)
    return np_data


def get_robust_stats(path):
    data = pd.read_json(path)
    np_data = data[["band_1", "band_2"]].as_matrix()
    band_1 = []
    band_2 = []
    for i in range(np_data.shape[0]):
        el_1 = np_data[i, 0]
        el_2 = np_data[i, 1]
        band_1.append(el_1)
        band_2.append(el_2)

    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    median_1 = np.median(band_1)
    median_2 = np.median(band_2)
    per25_1 = np.percentile(band_1, 25)
    per25_2 = np.percentile(band_2, 25)
    per75_1 = np.percentile(band_1, 75)
    per75_2 = np.percentile(band_2, 75)
    res = {"med1": median_1, "med2": median_2, "q1_1": per25_1, "q1_2": per25_2, "q3_1": per75_1, "q3_2": per75_2}
    print(res)


if __name__ == "__main__":
    original = "../data/orig/train.json"
    # res = split(original, n_splits=4)
    # for i, (tr, tt) in enumerate(res):
    #     np.save("../data/folds/train_%s" % i, tr)
    #     np.save("../data/folds/test_%s" % i, tt)
    # get_mu_sigma(original)
    # get_robust_stats(original)
    simple_split("../data/encoder_train.npy")
    print("Finished!")
