from torch import nn
from base.model import BaseAutoEncoder


class IcebergEncoder(BaseAutoEncoder):
    def __init__(self, feature_planes, encoder_only=False, fold_number=None):
        positional = [feature_planes]
        named = {"encoder_only": encoder_only, "fold_number": fold_number}
        super().__init__(pos_params=positional, named_params=named, model_name="AutoEncoder",
                         best_model_name="./models/aenc_best.mdl")
        self.fold_number = fold_number
        self._encoder_shape = True
        self._encoder_only = encoder_only

        input_shape = 8 * 17 * 17
        self.fc1 = nn.Linear(input_shape, 256, bias=True)
        self.fc2 = nn.Linear(256, 32, bias=True)

        self.fc_02 = nn.Linear(32, 256, bias=False)
        self.fc_01 = nn.Linear(256, input_shape, bias=False)

        self.fc_02.weight = self.fc2.weight
        self.fc_01.weight = self.fc1.weight

        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)

        self.activation = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(feature_planes, 16, kernel_size=3, padding=0),
            self.activation,
            self.pooling,
            nn.Conv2d(16, 8, kernel_size=3, padding=0),
            self.activation,
            self.pooling,
        )

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(16, feature_planes, kernel_size=3, stride=2, output_padding=1),
            nn.Upsample(size=(75, 75), mode="bilinear"),
            self.activation
        )

    def decode(self, x):
        if self._encoder_shape is True:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        out = self.fc_02(x)
        out = self.tanh(out)
        out = self.fc_01(out)

        out = out.view(out.size(0), 8, 17, 17)
        out = self.conv_decoder(out)
        self._encoder_shape = False
        return out

    def encode(self, x):
        if self._encoder_shape is False:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        enc = self.conv_encoder(x)  # shape N * 8 * 17 * 17

        enc = enc.view(enc.size(0), -1)
        enc = self.fc1(enc)
        enc = self.tanh(enc)
        enc = self.fc2(enc)
        enc = self.tanh(enc)
        self._encoder_shape = True
        return enc

    def forward(self, x):
        enc = self.encode(x)
        if self._encoder_only:
            return enc
        dec = self.decode(enc)
        return dec


class VariationalAutoEncoder(BaseAutoEncoder):
    def __init__(self, feature_planes, encoder_only=False, fold_number=None):
        positional = [feature_planes]
        named = {"encoder_only": encoder_only, "fold_number": fold_number}
        super().__init__(pos_params=positional, named_params=named, model_name="AutoEncoder",
                         best_model_name="./models/var_aenc_best.mdl")
        self.fold_number = fold_number
        self._encoder_shape = True
        self._encoder_only = encoder_only
        input_size = 2 * 75 * 75

        self.activation = nn.Tanh()
        self.fc1 = nn.Linear(input_size, 256, bias=True)
        self.fc2 = nn.Linear(256, 32, bias=True)
        nn.init.xavier_normal(self.fc1.weight, gain=1)
        nn.init.xavier_normal(self.fc2.weight, gain=1)
        self.fc_02 = nn.Linear(32, 256, bias=True)
        self.fc_01 = nn.Linear(256, input_size, bias=True)

        self.fc_01.weight = self.fc1.weight
        self.fc_02.weight = self.fc2.weight

    def decode(self, x):
        if self._encoder_shape is True:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        out = self.fc_02(x)
        out = self.activation(out)
        out = self.fc_01(out)
        out = self.activation(out)
        self._encoder_shape = False
        return out

    def encode(self, x):
        if self._encoder_shape is False:
            self.fc1.weight.data.t_()
            self.fc2.weight.data.t_()
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        self._encoder_shape = True
        return out

    def forward(self, x):
        enc = self.encode(x)
        if self._encoder_only:
            return enc
        dec = self.decode(enc)
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
