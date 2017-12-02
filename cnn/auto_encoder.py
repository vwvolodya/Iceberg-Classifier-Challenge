from torch import nn
from base.model import BaseAutoEncoder


class IcebergEncoder(BaseAutoEncoder):
    def __init__(self, feature_planes, encoder_only=False, fold_number=None):
        positional = [feature_planes]
        named = {"encoder_only": encoder_only, "fold_number": fold_number}
        super().__init__(pos_params=positional, named_params=named, model_name="AutoEncoder",
                         best_model_name="./models/aenc_best.mdl")
        self.fold_number = fold_number
        self.activation = nn.ELU(inplace=True)
        self._encoder_only = encoder_only
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_planes, 16, kernel_size=3, padding=0),
            self.activation,
            self.pooling,
            nn.Conv2d(16, 12, kernel_size=3, padding=0),
            self.activation,
            self.pooling,
            nn.Conv2d(12, 8, kernel_size=3, padding=0),
            self.activation,
            self.pooling
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 12, kernel_size=3, stride=2, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(12, 16, kernel_size=3, stride=2, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(16, feature_planes, kernel_size=3, stride=2, output_padding=1),
            nn.Upsample(size=(75, 75), mode="bilinear")
        )

    def forward(self, x):
        enc = self.encoder(x)
        if self._encoder_only:
            return enc
        dec = self.decoder(enc)
        return dec


if __name__ == "__main__":
    def dummy():
        import torch
        from base.dataset import DummyDataset
        from torch.utils.data import DataLoader
        from torch.optim import Adam
        from base.logger import Logger

        val = torch.randn((4, 2, 75, 75))
        ds = DummyDataset((2, 75, 75), (2, 75, 75), 10)
        loader = DataLoader(ds, batch_size=4)
        val_loader = DataLoader(ds, batch_size=2)
        model = IcebergEncoder(2)
        val = IcebergEncoder.to_var(val, use_gpu=False)
        result = model.predict(val)
        opt = Adam(model.parameters())
        loss_fn = nn.MSELoss()
        if torch.cuda.is_available():
            model.cuda()
            loss_fn.cuda()
        log = Logger("/tmp", erase_folder_content=False)

        losses = model.fit(opt, loss_fn, loader, val_loader, 10, log)
        print(losses)

    dummy()
    print("Done!")
