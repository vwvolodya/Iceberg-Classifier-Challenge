import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm as progressbar
from torchvision import transforms
from scipy import ndimage, signal
from base.dataset import BaseDataset, ToTensor
from base.exceptions import ProjectException


# scaler params computed on dataset
MIN_MAX = {"min1": -45.594448, "min2": -45.655499, "max1": 34.574917, "max2": 20.154249}
MU_SIGMA = {"mu1": -20.655831, "mu2": -26.320702, "sigma1": 5.200838, "sigma2": 3.395518}
MED_Q = {'med1': -21.0596, 'med2': -26.3451, 'q1_1': -24.1442, 'q1_2': -28.3731, 'q3_1': -17.5402, 'q3_2': -24.3858}


class Rotate:
    def __init__(self, angle, rnd=False):
        self.standard_angle = angle in [90, 180, 270]
        self.angle = angle
        self.rnd = rnd

    def __call__(self, item):
        image = item["inputs"]
        channels = image.shape[0]
        planes = [image[i, :, :] for i in range(channels)]
        if not self.standard_angle:
            images = [ndimage.rotate(im, self.angle, axes=(1, 0), mode="nearest", reshape=False) for im in planes]
        else:
            if self.angle == 90:
                images = [np.rot90(im) for im in planes]
            elif self.angle == 180:
                images = [np.rot90(np.rot90(im)) for im in planes]
            elif self.angle == 270:
                images = [np.rot90(np.rot90(np.rot90(im))) for im in planes]
            else:
                raise ProjectException("Invalid angle!")
        image = np.stack(images, axis=0)
        item["inputs"] = image
        return item


class Flip:
    def __init__(self, axis=2, rnd=False):
        self.axis = axis
        self.rnd = rnd

    def __call__(self, item):
        image = item["inputs"]
        item["inputs"] = np.flip(image, axis=self.axis).copy()
        return item


class IcebergDataset(BaseDataset):
    def __init__(self, path, inference_only=False, transform=None, im_dir=None, colormap="inferno",
                 top=None, mu_sigma=MU_SIGMA, denoise=False, add_feature_planes="no", width=75, return_angle=False):
        # add_feature_planes: one of "no", "simple", "complex"
        self.add_feature_planes = add_feature_planes
        self.return_angle = return_angle
        self.transform = transform
        self.denoise = denoise
        self.colormap = colormap
        self.mu_sigma = mu_sigma
        self.inference_only = inference_only
        self.im_dir = im_dir
        self.width = width  # according to dataset each "picture" is unrolled 75 * 75 "image"
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
        print_string = "Ds length %s \t" % self.data.shape[0]
        if not inference_only:
            print_string += "Positive %s" % sum(self.y)
        print(print_string)
        self.num_feature_planes = 2     # by default

    def __len__(self):
        return self.data.shape[0]

    @classmethod
    def get_image_stat(cls, image):
        mean_1 = np.mean(image)
        std_1 = np.std(image)
        median_1 = np.median(image)
        maximum = np.max(image)
        minimum = np.min(image)
        percentile_75 = np.percentile(image, 75)
        return mean_1, std_1, median_1, maximum, minimum, percentile_75

    def _get_image(self, idx):
        ch_1 = self.ch1[idx]
        ch_2 = self.ch2[idx]
        ch1_2d = np.reshape(ch_1, (self.width, self.width))
        ch2_2d = np.reshape(ch_2, (self.width, self.width))
        angle = self.angle[idx]
        if not isinstance(angle, float):
            angle = 39.26   # mean angle = 39.26
        multiplier = np.cos(np.deg2rad(angle))
        if self.denoise:
            ch1_2d = self._denoise(ch1_2d)
            ch2_2d = self._denoise(ch2_2d)
        if self.mu_sigma is not None:
            # ch1_2d = (ch1_2d - self.mu_sigma["mu1"]) / (self.mu_sigma["sigma1"]) * multiplier
            # ch2_2d = (ch2_2d - self.mu_sigma["mu2"]) / (self.mu_sigma["sigma2"]) * multiplier
            ch1_2d = (ch1_2d - MED_Q["med1"]) / (MED_Q["q3_1"] - MED_Q["q1_1"]) * multiplier / 3
            ch2_2d = (ch2_2d - MED_Q["med2"]) / (MED_Q["q3_2"] - MED_Q["q1_2"]) * multiplier / 3
        image = np.stack((ch1_2d, ch2_2d), axis=0)  # PyTorch uses NCHW ordering
        return image

    @classmethod
    def __correlate(cls, im1, im2):
        # images here should have 3d shape
        im1_gray = np.sum(im1, axis=0)
        im2_gray = np.sum(im2, axis=0)

        im1_gray -= np.mean(im1_gray)
        im2_gray -= np.mean(im2_gray)

        corr = signal.correlate(im1_gray, im2_gray, mode='same')
        return corr

    @classmethod
    def __fft_convolve(cls, im1, im2):
        # images here should have 3d shape
        im1_gray = np.sum(im1, axis=0)
        im2_gray = np.sum(im2, axis=0)

        im1_gray -= np.mean(im1_gray)
        im2_gray -= np.mean(im2_gray)

        fft = signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')
        return fft

    @classmethod
    def __multiply(cls, im1, im2):
        image = np.multiply(im1, im2)
        return image

    def _get_simple_planes(self, image):
        plane_1 = image[0, :, :]
        plane_2 = image[1, :, :]
        plane_3 = plane_1 + plane_2
        band_1 = (plane_1 - plane_1.mean()) / (plane_1.max() - plane_1.min())
        band_2 = (plane_2 - plane_2.mean()) / (plane_2.max() - plane_2.min())
        band_3 = (plane_3 - plane_3.mean()) / (plane_3.max() - plane_3.min())

        image = np.stack((band_1, band_2, band_3), axis=0)
        self.num_feature_planes = image.shape[0]
        return image

    def _add_planes(self, image):
        plane_1 = image[0, :, :]
        plane_2 = image[1, :, :]
        averaged = np.mean(image, axis=0)
        gauss3 = self._denoise(averaged, algo="gauss")
        median_3 = self._denoise(averaged, algo="median")

        multiplied = self.__multiply(plane_1, plane_2)
        gauss_multiplied = self._denoise(multiplied, algo="gauss")
        median_multiplied = self._denoise(multiplied, algo="median")

        correlated = self.__correlate(image, image)
        correlated = correlated / np.abs(np.max(correlated))
        image = np.stack((plane_1, plane_2, median_3, gauss3, correlated, median_multiplied, gauss_multiplied), axis=0)

        self.num_feature_planes = image.shape[0]
        return image

    def __getitem__(self, idx):
        image = self._get_image(idx)
        if self.add_feature_planes == "complex":
            image = self._add_planes(image)
        elif self.add_feature_planes == "simple":
            image = self._get_simple_planes(image)
        item = {"inputs": image}
        if self.return_angle:
            item["angle"] = np.array([self.angle[idx]])
        if self.inference_only:
            y = np.array([0])
            item["id"] = self.ids[idx]
        else:
            y = np.array([self.y[idx]])
        item["targets"] = y
        if self.transform:
            item = self.transform(item)
        return item

    def _make_heatmap(self, image, path, label=None, angle=None, **kwargs):
        string = ""
        for k, v in kwargs.items():
            string += str(k) + " " + "{0:.2f}".format(v) + " "
        if label is not None:
            plt.suptitle("%s, angle: %s %s" % (str(label), str(angle), string))
        plt.imshow(image, cmap=self.colormap)
        plt.savefig(path)
        plt.clf()

    def _denoise(self, image, algo="gauss"):
        if algo == "gauss":
            new_image = ndimage.gaussian_filter(image, 2)
        elif algo == "median":
            new_image = ndimage.median_filter(image, 3)
        else:
            raise Exception("Unknown algorithm. Use one of (gauss, median)")
        return new_image

    def vis(self, idx, average=False, prefix=""):
        base_dir = self.im_dir or "./"
        image = self[idx]["inputs"]
        if not isinstance(image, np.ndarray):
            image = image.numpy()
        if not self.inference_only:
            label = self.y[idx]
        else:
            label = "Unk"
        angle = self.angle[idx]
        if average:
            image = np.mean(image, axis=0)
            mu, sigma, median, maximum, minimum, percentile75 = self.get_image_stat(image)
            path = os.path.join(base_dir, "%sim_%s_%s.jpg" % (prefix, idx, label))
            self._make_heatmap(image, path, label=label, angle=angle, mu=mu, sigma=sigma, med=median, per75=percentile75)
        else:
            channels = image.shape[0]
            paths = [os.path.join(base_dir, "%sim_%s_ch%s_%s.jpg" % (prefix, idx, i, label)) for i in range(channels)]
            image_planes = [image[i, :, :] for i in range(channels)]
            mu_sigmas = [self.get_image_stat(i) for i in image_planes]
            for c in range(channels):
                self._make_heatmap(image_planes[c], paths[c], label=label, angle=angle,
                                   mu=mu_sigmas[c][0], sigma=mu_sigmas[c][1], med=mu_sigmas[c][2])


def _test_set():
    ds = IcebergDataset("../data/orig/test.json", im_dir="../data/vis/test", inference_only=True,
                        mu_sigma=None, colormap="inferno", add_feature_planes="complex")
    for i in progressbar(range(len(ds))):
        # print(i, ds[i]["inputs"].size(), ds[i]["id"])
        ds.vis(i, average=False, prefix="pure_")
        if i == 3:
            break

    loader = DataLoader(ds, batch_size=6, shuffle=False, num_workers=1)
    for i, batch in enumerate(loader):
        print(i, batch["inputs"].size(), batch["id"])
        if i == 3:
            break


if __name__ == "__main__":
    from torch.utils.data import DataLoader


    def train_set():
        t1 = ToTensor()
        t2 = transforms.Compose([Flip(axis=2), ToTensor()])
        t3 = transforms.Compose([Flip(axis=1), ToTensor()])
        t4 = transforms.Compose([Flip(axis=2), Flip(axis=1), ToTensor()])
        t5 = transforms.Compose([Flip(axis=1), Flip(axis=2), ToTensor()])
        t6 = transforms.Compose([Rotate(90), ToTensor()])

        ds1 = IcebergDataset("../data/all.npy", transform=None, im_dir="../data/vis/train",
                             colormap="inferno", add_feature_planes="complex")
        for i in progressbar(range(len(ds1))):
            sample = ds1[i]
            ds1.vis(i, average=False, prefix="")
            # print(i, sample['inputs'].size(), sample['targets'].size(), sample["targets"].numpy()[0])
            # if i == 10:
            #     break

        dataloader = DataLoader(ds1, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch, sample_batched['inputs'].size(), sample_batched['targets'].size())
            if i_batch == 3:
                break

    train_set()
    # test_set()
