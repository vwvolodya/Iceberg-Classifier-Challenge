from cnn.dataset import IcebergDataset, ToTensor
from cnn.model import ResNet, BasicBlock, LeNet
from cnn.inception import Inception
from torch.utils.data import DataLoader
from tqdm import tqdm as progressbar
from collections import defaultdict
import pandas as pd
import numpy as np
import torch


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def infer(path, num_folds, losses, average=True):
    ds = IcebergDataset(path, inference_only=True, transform=ToTensor(), add_feature_planes="complex")
    loader = DataLoader(ds, 64)
    predictions = defaultdict(list)
    weights = [1-i for i in losses]
    a = softmax(np.array(weights))

    for fold in range(num_folds):
        model = LeNet.restore("./models/LeNet_19_fold_None.mdl")
        if torch.cuda.is_available():
            model.cuda()
        iterator = iter(loader)
        iter_per_epoch = len(loader)
        for _ in progressbar(range(iter_per_epoch)):
            next_batch = next(iterator)
            inputs_tensor, ids = next_batch["inputs"], next_batch["id"]
            inputs = model.to_var(inputs_tensor)
            probs, _ = model.predict(inputs, return_classes=False)
            probs = model.to_np(probs).squeeze()
            probs = probs.tolist()
            chunk = dict(zip(ids, probs))
            for k, v in chunk.items():
                predictions[k].append(v)
    if average:
        result = {k: sum(v) / len(v) for k, v in predictions.items()}
    else:
        result = {}
        for k, v in predictions.items():
            prob = np.mean(np.array(v))
            if prob <= 0.1:
                prob = 0
            elif prob >= 0.9:
                prob = 1
            result[k] = prob
    return result


if __name__ == "__main__":
    original = "../data/orig/test.json"
    total_folds = 1
    scores = [0.25787729311447877, 0.24661139914622673, 0.25538152341659254, 0.31257020510160005]
    data = infer(original, total_folds, scores, average=False)

    new_df = pd.DataFrame(list(data.items()), columns=["id", "is_iceberg"])
    new_df.to_csv("../data/predicted1.csv", float_format='%.6f', index=False)
    print(new_df.shape)
