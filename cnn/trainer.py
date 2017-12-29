import signal
import pandas as pd
import torch
from torch import nn
from base.logger import Logger
from torch.nn import functional as F
from cnn.dataset import IcebergDataset, ToTensor, Flip, Rotate, Ravel
from cnn.model import LeNet
from cnn.inception import Inception
from cnn.auto_encoder import VariationalAutoEncoder
from cnn.aenc_dataset import AutoEncoderDataset
from torch.utils.data import DataLoader, ConcatDataset
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
        net = self.model_class(self.num_feature_planes, config["conv"], config["fc1"], momentum=config["momentum"],
                               fold_number=None, gain=config["gain"], model_prefix="final_")

        if torch.cuda.is_available():
            net.cuda()
            self.loss_func.cuda()
        big_train_set = IcebergDataset("../data/all.npy", transform=transformations, top=self.train_top,
                                       add_feature_planes="no")
        val_ds = IcebergDataset("../data/folds/test_3.npy", transform=ToTensor(),
                                top=self.test_top, add_feature_planes="no")

        train_loader = DataLoader(big_train_set, batch_size=config["train_batch_size"], num_workers=12,
                                  pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["test_batch_size"], num_workers=6, pin_memory=True)

        optim = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["lambda"])
        best = net.fit(optim, self.loss_func, train_loader, val_loader, epochs, logger=main_logger)
        print()
        print("Best was ", best)
        return best

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
            net = self.model_class(self.num_feature_planes, config["conv"], config["fc1"],
                                   momentum=config["momentum"], fold_number=f, gain=config["gain"],
                                   model_prefix=model_prefix)

            if torch.cuda.is_available():
                net.cuda()
                self.loss_func.cuda()
            train_set = IcebergDataset("../data/folds/train_%s.npy" % f, transform=transformations,
                                       top=self.train_top, add_feature_planes="no")

            val_ds = IcebergDataset("../data/folds/test_%s.npy" % f, transform=ToTensor(),
                                    top=self.test_top, add_feature_planes="no")

            train_loader = DataLoader(train_set, batch_size=config["train_batch_size"], num_workers=12,
                                      pin_memory=True, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config["test_batch_size"], num_workers=6, pin_memory=True)

            optim = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["lambda"])
            best = net.fit(optim, self.loss_func, train_loader, val_loader, epochs, logger=main_logger)
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


def _train_classifiers():
    parameter_grid = {
        "lr": [0.0001, 0.0003],
        "gain": [0.1],
        "conv": [48, 66],
        "lambda": [0.05, 0.003, 0.001],
        "momentum": [0.5, 0.25, 0.1, 0.05, 0.125],
        "test_batch_size": [64],
        "train_batch_size": [64, 128, 192],
        "fc1": [64, 128],
    }
    best_config = {
        "lr": 0.0001, "gain": 0.1, "conv": (16, 24, 24, 16), "lambda": 0.01, "fc1": 16,
        "fc2": 256, "train_batch_size": 128, "test_batch_size": 64, "momentum": 0.5
    }
    best_config_inception = {
        "lr": 0.0001, "gain": 0.1, "conv": 48, "lambda": 0.001, "fc1": 64, "train_batch_size": 192,
        "test_batch_size": 64, "momentum": 0.1
    }
    # best_config_inception = {
    #     "lr": 0.0001, "gain": 0.1, "conv": 48, "lambda": 0.05, "fc1": 64, "train_batch_size": 128,
    #     "test_batch_size": 64, "momentum": 0.125   # config scored the best result
    # }
    n_folds = 4
    top = None
    val_top = None
    num_planes = 2

    one_transform = transforms.Compose(
        [
            Flip(axis=2, rnd=True),
            Flip(axis=1, rnd=True),
            Rotate(90, rnd=True),
            Rotate(180, rnd=True),
            ToTensor()
        ]
    )

    loss_func = nn.BCELoss()

    trainer = ModelTrainer(num_planes, LeNet, loss_func, n_folds, Logger, train_top=top, test_top=val_top)
    # loss_scores = trainer.random_search(100, parameter_grid, train_epochs=100, transformations=one_transform)
    # loss_scores = trainer.train_one_configuration(best_config, 100, one_transform)
    loss_scores = trainer.train_all(best_config, 100, one_transform)
    print(loss_scores)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    batch_size = x.size(0)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 2 * 75 * 75

    return BCE + KLD


def _train_auto_encoders():
    n_folds = 1
    top = None
    val_top = None
    train_bsize = 256
    test_b_size = 64
    num_planes = 2
    scores = []

    one_transform = transforms.Compose(
        [
            Flip(axis=2, targets_also=True, rnd=True),
            Flip(axis=1, targets_also=True, rnd=True),
            Rotate(90, targets_also=True, rnd=True),
            Rotate(180, targets_also=True, rnd=True),
            Ravel(),
            ToTensor()
        ]
    )
    val_transform = transforms.Compose([
        Ravel(),
        ToTensor()
    ])

    for f in range(n_folds):
        main_logger = Logger("../logs/enc/", erase_folder_content=False)
        train_set = AutoEncoderDataset("../data/all.npy", transform=one_transform, top=top)
        train_loader = DataLoader(train_set, batch_size=train_bsize, num_workers=12, pin_memory=True, shuffle=True)
        val_set = AutoEncoderDataset("../data/folds/test_1.npy", transform=val_transform, top=val_top)
        val_loader = DataLoader(val_set, batch_size=test_b_size, num_workers=6, pin_memory=True)

        encoder = VariationalAutoEncoder(num_planes, fold_number=None)
        if torch.cuda.is_available():
            encoder.cuda()
        optim = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0)
        best = encoder.fit(optim, loss_function, train_loader, val_loader, 100, logger=main_logger)
        scores.append(best)
        print()
        print("Best was ", best)
    print(scores)
    return scores


if __name__ == "__main__":
    _train_classifiers()
    print("Done!")
