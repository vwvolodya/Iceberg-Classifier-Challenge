from torch import nn
from base.model import BaseBinaryClassifier


class SimpleMLP(BaseBinaryClassifier):
    def __init__(self, model_prefix="", fold_number=None):
        named = {"fold_number": fold_number,
                 "model_prefix": model_prefix}
        super().__init__(pos_params=[], named_params=named, model_name="MLP",
                         best_model_name="./models/%s_best_%s.mdl" % (model_prefix, fold_number))
        self.fold_number = fold_number
        self.activation = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

        nn.init.xavier_normal(self.fc1.weight, gain=0.1)
        nn.init.xavier_normal(self.fc2.weight, gain=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    def train():
        from cnn.simple_ds import SimpleIcebergDataset
        from cnn.dataset import IcebergDataset, Ravel, ToTensor
        from torch.utils.data import DataLoader
        import torch
        from base.logger import Logger

        main_logger = Logger("../logs/simple", erase_folder_content=True)

        ds = SimpleIcebergDataset("../data/for_train_encoder.npy", transform=ToTensor())
        val_ds = SimpleIcebergDataset("../data/for_test_encoder.npy", transform=ToTensor())
        ldr = DataLoader(ds, batch_size=64)
        val_ldr = DataLoader(val_ds, batch_size=64)
        model = SimpleMLP()

        optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
        loss_fn = nn.BCELoss()
        if torch.cuda.is_available():
            model.cuda()
            loss_fn.cuda()
        best = model.fit(optim, loss_fn, ldr, val_ldr, 250, logger=main_logger)
        print(best)

    train()
    print("Done!")
