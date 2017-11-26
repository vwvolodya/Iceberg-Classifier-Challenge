import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class ToTensor:
    def __init__(self, excluded_keys=("id",)):
        self.excluded = excluded_keys

    def __call__(self, x):
        result = {k: torch.from_numpy(v).float() for k, v in x.items() if k not in self.excluded}
        for k in self.excluded:
            if k in x.keys():
                result[k] = x[k]
        return result
