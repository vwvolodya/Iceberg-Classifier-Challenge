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
    ds = IcebergDataset(path, inference_only=True, transform=ToTensor(), add_feature_planes=True)
    loader = DataLoader(ds, 64)
    predictions = defaultdict(list)
    weights = [1-i for i in losses]
    a = softmax(np.array(weights))

    for fold in range(num_folds):
        model = Inception(5, 32, None, None, num_classes=1, fold_number=fold)
        model.load("./models/clf_43_fold_0.mdl")
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
        result = {k: sum(np.array(v) * a) for k, v in predictions.items()}
    return result


if __name__ == "__main__":
    original = "../data/orig/test.json"
    total_folds = 1
    scores = [0.19850767403849545, 0.17849749947587648, 0.22, 0.32, 0.34]
    data = infer(original, total_folds, scores)

    new_df = pd.DataFrame(list(data.items()), columns=["id", "is_iceberg"])
    new_df.to_csv("../data/predicted.csv", float_format='%.4f', index=False)
    print(new_df.shape)
