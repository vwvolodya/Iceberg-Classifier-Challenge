import numpy as np
from base.dataset import BaseDataset


class SimpleIcebergDataset(BaseDataset):
    def __init__(self, path, inference_only=False, transform=None):
        data = np.load(path)
        self.transform = transform
        if not inference_only:
            self.x = data[:, : -1]
            self.y = data[:, -1]
        else:
            self.x = data

    def __len__(self):
        length = self.x.shape[0]
        return length

    def __getitem__(self, idx):
        y = np.array([self.y[idx]])
        x = self.x[idx, :]
        item = {"targets": y, "inputs": x}
        if self.transform:
            item = self.transform(item)
        return item
