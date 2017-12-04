import numpy as np
from tqdm import tqdm as progressbar
from cnn.dataset import IcebergDataset


class AutoEncoderDataset(IcebergDataset):
    def __getitem__(self, idx):
        image = self._get_image(idx)
        if self.add_feature_planes == "complex":
            image = self._add_planes(image)
        elif self.add_feature_planes == "simple":
            image = self._get_simple_planes(image)
        noise_factor = 0.4
        planes = [image[i, :, :] for i in range(image.shape[0])]
        stats = [self.get_image_stat(i) for i in planes]
        masks = [np.random.binomial(1, 1 - noise_factor, i.shape) for i in planes]
        noise = [masks[i] * np.random.normal(loc=stats[i][0], scale=stats[i][1],
                                             size=masks[i].shape) for i in range(len(planes))]
        noisy = [planes[i] + noise[i] for i in range(len(planes))]
        noisy = np.stack(noisy, axis=0)
        item = {"inputs": noisy, "targets": image}
        if self.transform:
            item = self.transform(item)
        return item

    def vis(self, idx, average=False, prefix=""):
        base_dir = self.im_dir or "./"
        image1 = self[idx]["inputs"]
        image2 = self[idx]["targets"]
        self._vis_image(idx, image2, average, base_dir, prefix)
        self._vis_image(idx, image1, average, base_dir, "noise_" + prefix)


if __name__ == "__main__":
    ds1 = AutoEncoderDataset("../data/folds/test_0.npy", transform=None, im_dir="../data/vis/test",
                             colormap="inferno", add_feature_planes="no")
    for i in progressbar(range(len(ds1))):
        sample = ds1[i]
        ds1.vis(i, average=True, prefix="")
