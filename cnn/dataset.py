import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from base.dataset import BaseDataset, ToTensor


# min max scaler params computed on dataset. order: min1, min2, max1, max2
MIN_MAX = {"min1": -45.594448, "min2": -45.655499, "max1": 34.574917, "max2": 20.154249}


class IcebergDataset(BaseDataset):
    def __init__(self, path, inference_only=False, transform=None, augment=None, im_dir=None, min_max=MIN_MAX):
        self.transform = transform
        self.min_max = min_max
        self.inference_only = inference_only
        self.im_dir = im_dir
        if inference_only:
            self.ids = None
        self.data = np.load(path)
        self.ch1 = self.data[:, 0]
        self.ch2 = self.data[:, 1]
        self.angle = self.data[:, 2]
        self.y = self.data[:, -1]
        self.width = 75     # according to dataset each "picture" is unrolled 75 * 75 "image"

    def __len__(self):
        return self.data.shape[0]

    def _get_image(self, idx, scale=False):
        ch_1 = self.ch1[idx]
        ch_2 = self.ch2[idx]
        ch1_2d = np.reshape(ch_1, (self.width, self.width))
        ch2_2d = np.reshape(ch_2, (self.width, self.width))
        if scale:
            diff_1 = self.min_max["max1"] - self.min_max["min1"]
            diff_2 = self.min_max["max2"] - self.min_max["min2"]
            ch1_2d = (ch1_2d - self.min_max["min1"]) / diff_1
            ch2_2d = (ch2_2d - self.min_max["min2"]) / diff_2
        image = np.stack((ch1_2d, ch2_2d), axis=0)  # PyTorch uses NCHW ordering
        return image

    def __getitem__(self, idx):
        image = self._get_image(idx)
        item = {"inputs": image}
        if self.inference_only:
            y = np.array([0])
            item["id"] = np.array([self.ids[idx]])
        else:
            y = np.array([self.y[idx]])
        item["targets"] = y
        if self.transform:
            item = self.transform(item)
        return item

    @classmethod
    def _make_heatmap(cls, image, path, label=None):
        if label is not None:
            plt.suptitle("Is iceberg: %s" % str(label))
        plt.imshow(image, cmap="jet")
        plt.savefig(path)
        plt.clf()

    def vis(self, idx, average=True, scale=False):
        base_dir = self.im_dir or "./"
        image = self._get_image(idx, scale=scale)
        label = self.y[idx]
        if average:
            path = os.path.join(base_dir, "im_%s_%s.jpg" % (idx, label))
            image = np.mean(image, axis=0)
            self._make_heatmap(image, path, label=label)
        else:
            path_1 = os.path.join(base_dir, "im_%s_ch1_%s.jpg" % (idx, label))
            path_2 = os.path.join(base_dir, "im_%s_ch2_%s.jpg" % (idx, label))
            image_1 = image[0, :, :]
            image_2 = image[1, :, :]
            self._make_heatmap(image_1, path_1, label=label)
            self._make_heatmap(image_2, path_2, label=label)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = IcebergDataset("../data/folds/train_0.npy", transform=ToTensor(), im_dir="../data/vis")
    for i in range(len(ds)):
        sample = ds[i]
        ds.vis(i, average=False, scale=True)
        print(i, sample['inputs'].size(), sample['targets'].size(), sample["targets"].numpy()[0])
        if i == 100:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
        if i_batch == 3:
            break
