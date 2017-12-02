import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class DummyDataset(BaseDataset):
    def __init__(self, input_shape, output_shape, length):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        val_1 = torch.randn(self.input_shape)
        val_2 = torch.randn(self.output_shape)
        item = {"inputs": val_1, "targets": val_2}
        return item


class ToTensor:
    def __init__(self, excluded_keys=("id",)):
        self.excluded = excluded_keys

    def __call__(self, x):
        result = {k: torch.from_numpy(v).float() for k, v in x.items() if k not in self.excluded}
        for k in self.excluded:
            if k in x.keys():
                result[k] = x[k]
        return result
