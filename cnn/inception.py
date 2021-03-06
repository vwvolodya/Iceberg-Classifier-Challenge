import torch
from torch import nn
from base.model import BaseBinaryClassifier


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, gain, momentum, activation, **kwargs):
        super(BasicConv2d, self).__init__()
        activations = {
            "elu": nn.ELU(inplace=True),
            "relu": nn.ReLU(inplace=True),
            "lrelu": nn.LeakyReLU(0.1, inplace=True)
        }
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.activation = activations[activation]
        nn.init.xavier_normal(self.conv.weight, gain=gain)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class _InceptionA(nn.Module):
    def __init__(self, features, num_filters, momentum, gain, activation):
        super().__init__()
        self.branch_3x3 = BasicConv2d(features, num_filters // 2, kernel_size=3, padding=1,
                                      momentum=momentum, gain=gain, activation=activation)
        self.branch_5x5 = BasicConv2d(features, num_filters // 2, kernel_size=5, padding=2,
                                      momentum=momentum, gain=gain, activation=activation)

    def forward(self, x):
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)

        out = [branch_3x3, branch_5x5]
        out = torch.cat(out, 1)
        return out


class _InceptionC(nn.Module):
    def __init__(self, features, num_filters, momentum, gain, activation):
        super().__init__()
        assert num_filters % 3 == 0, "InceptionC must have num_filters that can be divided by 3!"
        self.branch_5x1 = BasicConv2d(features, num_filters // 3, kernel_size=(5, 1),
                                      padding=(2, 0), momentum=momentum, gain=gain, activation=activation)
        self.branch_1x5 = BasicConv2d(features, num_filters // 3, kernel_size=(1, 5),
                                      padding=(0, 2), momentum=momentum, gain=gain, activation=activation)
        self.branch_3x3 = BasicConv2d(features, num_filters // 3, kernel_size=3, padding=1,
                                      momentum=momentum, gain=gain, activation=activation)

    def forward(self, x):
        branch_5x1 = self.branch_5x1(x)
        branch_1x5 = self.branch_1x5(x)
        branch_3x3 = self.branch_3x3(x)

        out = [branch_5x1, branch_1x5, branch_3x3]
        out = torch.cat(out, 1)
        return out


class _InceptionE(nn.Module):
    def __init__(self, features, num_filters, momentum, gain, activation):
        super().__init__()
        assert num_filters % 3 == 0, "InceptionE must have num_filters that can be divided by 3!"
        self.branch_3x1 = BasicConv2d(features, num_filters // 3, kernel_size=(3, 1), padding=(1, 0),
                                      momentum=momentum, gain=gain, activation=activation)
        self.branch_1x3 = BasicConv2d(features, num_filters // 3, kernel_size=(1, 3), padding=(0, 1),
                                      momentum=momentum, gain=gain, activation=activation)
        self.branch_3x3 = BasicConv2d(features, num_filters // 3, kernel_size=3, padding=1,
                                      momentum=momentum, gain=gain, activation=activation)

    def forward(self, x):
        branch_3x1 = self.branch_3x1(x)
        branch_1x3 = self.branch_1x3(x)
        branch_3x3 = self.branch_3x3(x)

        out = [branch_3x1, branch_1x3, branch_3x3]
        out = torch.cat(out, 1)

        return out


class Inception(BaseBinaryClassifier):
    def __init__(self, num_feature_planes, inner, output_features, fc1, momentum=0.1, num_classes=1,
                 fold_number=0, gain=0.1, model_prefix=""):
        positional = [num_feature_planes, inner, output_features, fc1]
        named = {"num_classes": num_classes, "fold_number": fold_number, "gain": gain, "momentum": momentum}
        super().__init__(pos_params=positional, named_params=named, model_name="InceptionV",
                         best_model_name="./models/inception_%s_best_%s.mdl" % (model_prefix, fold_number))
        self.fold_number = fold_number
        self.activation = "elu"
        self.ac_func = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception_a = _InceptionA(num_feature_planes, inner, momentum=momentum,
                                       gain=gain, activation=self.activation)
        self.inception_e_0 = _InceptionE(inner, inner * 2, momentum=momentum,
                                         gain=gain, activation=self.activation)
        self.inception_e_1 = _InceptionE(inner * 2, inner * 2, momentum=momentum,
                                         gain=gain, activation=self.activation)
        self.inception_e_2 = _InceptionE(inner * 2, inner, momentum=momentum,
                                         gain=gain, activation=self.activation)

        self.fc1 = nn.Linear(inner * 4 * 4, fc1, bias=True)
        self.fc2 = nn.Linear(fc1, num_classes)
        nn.init.xavier_normal(self.fc1.weight, gain=gain)
        nn.init.xavier_normal(self.fc2.weight, gain=gain)

    def forward(self, x):
        out = self.inception_a(x)
        out = self.pool(out)
        out = self.inception_e_0(out)
        out = self.pool(out)
        out = self.inception_e_1(out)
        out = self.pool(out)
        out = self.inception_e_2(out)
        out = self.pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.ac_func(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    a = torch.randn(4, 5, 75, 75)
    inc = Inception(5, 24, None, 128)
    a = Inception.to_var(a, use_gpu=False)
    res = inc.predict(a, return_classes=True)
    print(res)
