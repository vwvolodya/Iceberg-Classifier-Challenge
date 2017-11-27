from cnn.dataset import IcebergDataset, ToTensor
from cnn.model import ResNet, BasicBlock, LeNet
from torch.utils.data import DataLoader
from tqdm import tqdm as progressbar
from collections import defaultdict
import pandas as pd
import numpy as np
import torch


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def infer(path, num_folds, losses):
    ds = IcebergDataset(path, inference_only=True, transform=ToTensor())
    loader = DataLoader(ds, 512)
    predictions = defaultdict(list)
    weights = [1-i for i in losses]
    a = softmax(np.array(weights))

    for fold in range(num_folds):
        # model = ResNet(BasicBlock, 2, [2, 2, 2, 2], num_classes=1, fold_number=fold)
        # model.load("./models/best_fold_%s.mdl" % fold)
        model = LeNet(num_classes=1, fold_number=fold)
        model.load("./models/le_best_fold_%s.mdl" % fold)
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

    result = {k: sum(np.array(v) * a) for k, v in predictions.items()}
    return result


if __name__ == "__main__":
    original = "../data/orig/test.json"
    total_folds = 4
    scores = [0.2756, 0.2966, 0.2740, 0.3988]
    data = infer(original, total_folds, scores)

    new_df = pd.DataFrame(list(data.items()), columns=["id", "is_iceberg"])
    new_df.to_csv("../data/predicted.csv", float_format='%.4f', index=False)
    print(new_df.shape)
