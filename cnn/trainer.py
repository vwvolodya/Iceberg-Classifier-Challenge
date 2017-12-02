import signal
import pandas as pd
import torch
from torch import nn
from base.logger import Logger
from cnn.dataset import IcebergDataset, ToTensor, Flip, Rotate
from cnn.model import LeNet
from cnn.inception import Inception
from torch.utils.data import DataLoader
from torchvision import transforms
from random import choice
from tqdm import tqdm as progressbar


class ModelTrainer:
    def __init__(self, num_feature_planes, model_class, loss_fn, num_folds, logger_class,
                 train_top=None, test_top=None):
        self.model_class = model_class
        self.loss_func = loss_fn
        self.num_folds = num_folds
        self.num_feature_planes = num_feature_planes
        self.logger_class = logger_class
        self.train_top = train_top
        self.test_top = test_top
        self._cache = {}        # used to keep track of tried configurations

        self._kill = False
        self._searching = False
        signal.signal(signal.SIGINT, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        print("Got signal number %s. Will stop as soon as possible!" % signum)
        if self._searching:
            self._kill = True
        else:
            print("Exiting now!")
            exit(2)

    def _get_random_config(self, config, verbose=True):
        generated = False
        res = None
        while not generated:
            res = {k: choice(v) for k, v in config.items()}
            key = hash(str(res))
            if key not in self._cache.keys():
                generated = True
                self._cache[key] = res
        if verbose:
            print("Next config, ", res)
        return res

    def train_all(self, config, epochs, transformations):
        main_logger = self.logger_class("../logs", erase_folder_content=False)
        net = LeNet(self.num_feature_planes, config["conv"], config["fc1"], None, fold_number=None,
                    gain=config["gain"], model_prefix="final")

        if torch.cuda.is_available():
            net.cuda()
            self.loss_func.cuda()
        train_sets = [
            IcebergDataset(
                "../data/all.npy", transform=t, top=self.train_top, add_feature_planes="complex"
            ) for t in transformations
        ]
        val_ds = IcebergDataset("../data/folds/test_3.npy", transform=ToTensor(),
                                top=self.test_top, add_feature_planes="complex")

        train_loaders = [DataLoader(ds, batch_size=config["train_batch_size"], num_workers=12,
                                    pin_memory=True, shuffle=True)
                         for ds in train_sets]
        val_loader = DataLoader(val_ds, batch_size=config["test_batch_size"], num_workers=6, pin_memory=True)

        optim = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["lambda"])
        best = net.fit(optim, self.loss_func, train_loaders, val_loader, epochs, logger=main_logger)
        print()
        print("Best was ", best)

    def train_one_configuration(self, config, epochs, transformations):
        assert "gain" in config
        assert "conv" in config
        assert "lr" in config
        assert "fc1" in config
        assert "lambda" in config
        assert "momentum" in config
        assert "train_batch_size" in config
        assert "test_batch_size" in config

        scores = []

        for f in range(self.num_folds):
            main_logger = self.logger_class("../logs/%s" % f, erase_folder_content=True)

            model_prefix = str(hash(str(config)))
            net = self.model_class(self.num_feature_planes, config["conv"], 16, config["fc1"],
                                   momentum=config["momentum"], fold_number=f, gain=config["gain"],
                                   model_prefix=model_prefix)

            if torch.cuda.is_available():
                net.cuda()
                self.loss_func.cuda()
            train_sets = [
                IcebergDataset(
                    "../data/folds/train_%s.npy" % f, transform=t, top=self.train_top, add_feature_planes="complex"
                ) for t in transformations
            ]
            val_ds = IcebergDataset("../data/folds/test_%s.npy" % f, transform=ToTensor(),
                                    top=self.test_top, add_feature_planes="complex")

            train_loaders = [DataLoader(ds, batch_size=config["train_batch_size"], num_workers=12,
                                        pin_memory=True, shuffle=True)
                             for ds in train_sets]
            val_loader = DataLoader(val_ds, batch_size=config["test_batch_size"], num_workers=6, pin_memory=True)

            optim = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["lambda"])
            best = net.fit(optim, self.loss_func, train_loaders, val_loader, epochs, logger=main_logger)
            print()
            print("Best was ", best)
            scores.append(best)
        return scores

    @classmethod
    def _save_results(cls, data, path):
        results = pd.DataFrame(data)
        results.to_csv(path, index=False)
        print("Results are saved to path %s" % path)

    def random_search(self, iterations, config, train_epochs, transformations,
                      verbose=True, result_path="../data/results.csv"):
        self._searching = True
        all_scores = []
        for _ in progressbar(range(iterations)):
            if self._kill:
                print("Will exit now because of signal!")
                break
            current_config = self._get_random_config(config)
            current_score = self.train_one_configuration(current_config, train_epochs, transformations)

            current_score.extend([str(current_config), str(hash(str(current_config)))])
            all_scores.append(current_score)
            if verbose:
                print(all_scores)
        self._save_results(all_scores, result_path)
        self._searching = False
        return all_scores


if __name__ == "__main__":
    parameter_grid = {
        "lr": [0.0001],
        "gain": [0.01],
        "conv": [(16, 24, 24, 16)],
        "lambda": [0, 0.003, 0.001],
        "momentum": [0.5, 0.25, 0.1, 0.05],
        "test_batch_size": [64],
        "train_batch_size": [64, 128, 192, 256, 384],
        "fc1": [16],
    }
    best_config = {
        "lr": 0.0001, "gain": 0.01, "conv": (16, 24, 24, 16), "lambda": 0, "fc1": 16,
        "fc2": 256, "train_batch_size": 512, "test_batch_size": 64, "momentum": 0.5
    }
    best_config_inception = {
        "lr": 0.0001, "gain": 0.01, "conv": 24, "lambda": 0, "fc1": 16, "train_batch_size": 256,
        "test_batch_size": 64, "momentum": 0.125
    }
    n_folds = 4
    top = None
    val_top = None
    train_bsize = 384
    test_b_size = 64
    num_planes = 7

    t1 = ToTensor()

    t2 = transforms.Compose([Flip(axis=2), ToTensor()])
    t3 = transforms.Compose([Flip(axis=1), ToTensor()])
    t4 = transforms.Compose([Flip(axis=2), Flip(axis=1), ToTensor()])

    t5 = transforms.Compose([Rotate(90), ToTensor()])
    t6 = transforms.Compose([Rotate(180), ToTensor()])
    t7 = transforms.Compose([Rotate(270), ToTensor()])

    t8 = transforms.Compose([Rotate(90), Flip(axis=1), ToTensor()])
    t9 = transforms.Compose([Rotate(90), Flip(axis=2), ToTensor()])
    t10 = transforms.Compose([Rotate(90), Flip(axis=1), Flip(axis=1), ToTensor()])

    t11 = transforms.Compose([Rotate(180), Flip(axis=1), ToTensor()])
    t12 = transforms.Compose([Rotate(180), Flip(axis=2), ToTensor()])
    t13 = transforms.Compose([Rotate(180), Flip(axis=1), Flip(axis=1), ToTensor()])

    t14 = transforms.Compose([Rotate(270), Flip(axis=1), ToTensor()])
    t15 = transforms.Compose([Rotate(270), Flip(axis=2), ToTensor()])
    t16 = transforms.Compose([Rotate(270), Flip(axis=1), Flip(axis=1), ToTensor()])

    all_transformations = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16]

    loss_func = nn.BCELoss()

    trainer = ModelTrainer(num_planes, Inception, loss_func, n_folds, Logger, train_top=top, test_top=val_top)
    # loss_scores = trainer.random_search(25, parameter_grid, train_epochs=20, transformations=all_transformations)
    loss_scores = trainer.train_one_configuration(best_config_inception, 40, all_transformations)
    # trainer.train_all(best_config, 30, all_transformations)
    print(loss_scores)