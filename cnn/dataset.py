import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy import ndimage
from base.dataset import BaseDataset, ToTensor


# scaler params computed on dataset
MIN_MAX = {"min1": -45.594448, "min2": -45.655499, "max1": 34.574917, "max2": 20.154249}
MU_SIGMA = {"mu1": -20.655831, "mu2": -26.320702, "sigma1": 5.200838, "sigma2": 3.395518}


class Rotate:
    def __init__(self, angle):
        self.standard_angle = angle in [90, 180, 270]
        self.angle = angle

    def __call__(self, item):
        image = item["inputs"]
        plane_1 = image[0, :, :]
        plane_2 = image[1, :, :]
        if not self.standard_angle:
            image_1 = ndimage.rotate(plane_1, self.angle, axes=(1, 0), mode="nearest", reshape=False)
            image_2 = ndimage.rotate(plane_2, self.angle, axes=(1, 0), mode="nearest", reshape=False)
        else:
            if self.angle == 90:
                image_1 = np.rot90(plane_1)
                image_2 = np.rot90(plane_2)
            elif self.angle == 180:
                image_1 = np.rot90(np.rot90(plane_1))
                image_2 = np.rot90(np.rot90(plane_2))
            elif self.angle == 270:
                image_1 = np.rot90(np.rot90(np.rot90(plane_1)))
                image_2 = np.rot90(np.rot90(np.rot90(plane_2)))
            else:
                raise Exception("Invalid angle!")
        image = np.stack((image_1, image_2), axis=0)
        item["inputs"] = image
        return item


class Flip:
    def __init__(self, axis=2):
        self.axis = axis

    def __call__(self, item):
        image = item["inputs"]
        item["inputs"] = np.flip(image, axis=self.axis).copy()
        return item


class IcebergDataset(BaseDataset):
    def __init__(self, path, inference_only=False, transform=None, im_dir=None,
                 min_max=MIN_MAX, top=None, mu_sigma=MU_SIGMA, denoise=False):
        self.transform = transform
        self.denoise = denoise
        self.min_max = min_max
        self.mu_sigma = mu_sigma
        self.inference_only = inference_only
        self.im_dir = im_dir
        self.width = 75  # according to dataset each "picture" is unrolled 75 * 75 "image"
        if inference_only:
            data = pd.read_json(path)
            self.data = data[["band_1", "band_2", "inc_angle"]].as_matrix()
            self.ids = data["id"].tolist()
        else:
            self.data = np.load(path)
            if top:
                self.data = self.data[:top, :]
            self.y = self.data[:, -1]
        self.ch1 = self.data[:, 0]
        self.ch2 = self.data[:, 1]
        self.angle = self.data[:, 2]
        print("Ds length ", self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    @classmethod
    def _get_image_stat(cls, image):
        mean_1 = np.mean(image)
        std_1 = np.std(image)
        return mean_1, std_1

    def _get_image(self, idx, scale=True):
        ch_1 = self.ch1[idx]
        ch_2 = self.ch2[idx]
        ch1_2d = np.reshape(ch_1, (self.width, self.width))
        ch2_2d = np.reshape(ch_2, (self.width, self.width))
        if self.denoise:
            ch1_2d = self._denoise(ch1_2d)
            ch2_2d = self._denoise(ch2_2d)
        mu1, sigma1 = self._get_image_stat(ch1_2d)
        mu2, sigma2 = self._get_image_stat(ch2_2d)
        if scale:
            # min max scaling
            # diff_1 = self.min_max["max1"] - self.min_max["min1"]
            # diff_2 = self.min_max["max2"] - self.min_max["min2"]
            # ch1_2d = (ch1_2d - self.min_max["min1"]) / diff_1
            # ch2_2d = (ch2_2d - self.min_max["min2"]) / diff_2

            ch1_2d = (ch1_2d - self.mu_sigma["mu1"]) / self.mu_sigma["sigma1"]
            ch2_2d = (ch2_2d - self.mu_sigma["mu2"]) / self.mu_sigma["sigma2"]
        image = np.stack((ch1_2d, ch2_2d), axis=0)  # PyTorch uses NCHW ordering
        return image, mu1, sigma1, mu2, sigma2

    def __getitem__(self, idx):
        image, _, _, _, _ = self._get_image(idx, scale=True)
        item = {"inputs": image}
        if self.inference_only:
            y = np.array([0])
            item["id"] = self.ids[idx]
        else:
            y = np.array([self.y[idx]])
        item["targets"] = y
        if self.transform:
            item = self.transform(item)
        return item

    @classmethod
    def _make_heatmap(cls, image, path, label=None, angle=None, **kwargs):
        string = ""
        for k, v in kwargs.items():
            string += str(k) + " " + "{0:.2f}".format(v) + " "
        if label is not None:
            plt.suptitle("%s, angle: %s %s" % (str(label), str(angle), string))
        plt.imshow(image, cmap="jet")
        plt.savefig(path)
        plt.clf()

    def _denoise(self, image):
        # new_image = ndimage.median_filter(image, 3)
        new_image = ndimage.gaussian_filter(image, 2)
        # new_image = ndimage.sobel(image, 2)
        return new_image

    def vis(self, idx, average=True, scale=False, denoise=False, prefix=""):
        base_dir = self.im_dir or "./"
        image, mu1, sigma1, mu2, sigma2 = self._get_image(idx, scale=scale)
        label = self.y[idx]
        angle = self.angle[idx]
        if average:
            image = np.mean(image, axis=0)
            mu, sigma = self._get_image_stat(image)
            path = os.path.join(base_dir, "%sim_%s_%s.jpg" % (prefix, idx, label))
            self._make_heatmap(image, path, label=label, angle=angle, mu=mu, sigma=sigma)
        else:
            path_1 = os.path.join(base_dir, "%sim_%s_ch1_%s.jpg" % (prefix, idx, label))
            path_2 = os.path.join(base_dir, "%sim_%s_ch2_%s.jpg" % (prefix, idx, label))
            image_1 = image[0, :, :]
            image_2 = image[1, :, :]
            if denoise:
                image_1 = self._denoise(image_1)
                image_2 = self._denoise(image_2)
            self._make_heatmap(image_1, path_1, label=label, angle=angle, mu1=mu1, sigma1=sigma1)
            self._make_heatmap(image_2, path_2, label=label, angle=angle, mu2=mu2, sigma2=sigma2)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    def train_set():
        t1 = ToTensor()
        t2 = transforms.Compose([Flip(axis=2), ToTensor()])
        t3 = transforms.Compose([Flip(axis=1), ToTensor()])
        t4 = transforms.Compose([Flip(axis=2), Flip(axis=1), ToTensor()])
        t5 = transforms.Compose([Flip(axis=1), Flip(axis=2), ToTensor()])
        t6 = transforms.Compose([Rotate(30), ToTensor()])

        ds1 = IcebergDataset("../data/folds/train_1.npy", transform=t6, im_dir="../data/vis")
        for i in range(len(ds1)):
            sample = ds1[i]
            ds1.vis(i, average=False, scale=True, denoise=False, prefix="rot_90_")
            print(i, sample['inputs'].size(), sample['targets'].size(), sample["targets"].numpy()[0])
            if i == 10:
                break

        dataloader = DataLoader(ds1, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
            if i_batch == 3:
                break

    def test_set():
        ds = IcebergDataset("../data/orig/test.json", transform=ToTensor(), im_dir="../data/vis", inference_only=True)
        for i in range(len(ds)):
            print(i, ds[i]["inputs"].size(), ds[i]["id"])
            if i == 3:
                break

        loader = DataLoader(ds, batch_size=6, shuffle=False, num_workers=1)
        for i, batch in enumerate(loader):
            print(i, batch["inputs"].size(), batch["id"])
            if i == 3:
                break

    train_set()
