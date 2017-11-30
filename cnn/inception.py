import torch
from torch import nn
from base.model import BaseBinaryClassifier


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, gain=0.01, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.activation = nn.ELU(inplace=True)
        nn.init.xavier_normal(self.conv.weight, gain=gain)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class _InceptionA(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_3x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x3 = BasicConv2d(num_filters, num_filters, kernel_size=3, padding=1)

        self.branch_5x5_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_5x5 = BasicConv2d(num_filters, num_filters, kernel_size=5, padding=2)

    def forward(self, x):
        branch_3x3_btlneck = self.branch_3x3_btlneck(x)
        branch_5x5_btlneck = self.branch_5x5_btlneck(x)

        branch_3x3 = self.branch_3x3(branch_3x3_btlneck)
        branch_5x5 = self.branch_5x5(branch_5x5_btlneck)

        out = [branch_3x3, branch_5x5]
        out = torch.cat(out, 1)
        return out


class _InceptionC(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_5x1_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_1x5_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)

        self.branch_5x1 = BasicConv2d(num_filters, num_filters, kernel_size=(5, 1), padding=(2, 0))
        self.branch_1x5 = BasicConv2d(num_filters, num_filters, kernel_size=(1, 5), padding=(0, 2))
        self.branch_3x3 = BasicConv2d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        branch_5x1_btlneck = self.branch_5x1_btlneck(x)
        branch_5x1 = self.branch_5x1(branch_5x1_btlneck)

        branch_1x5_btlneck = self.branch_1x5_btlneck(x)
        branch_1x5 = self.branch_1x5(branch_1x5_btlneck)

        branch_3x3_btlneck = self.branch_3x3_btlneck(x)
        branch_3x3 = self.branch_3x3(branch_3x3_btlneck)

        out = [branch_5x1, branch_1x5, branch_3x3]
        out = torch.cat(out, 1)
        return out


class _InceptionE(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_3x1_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x1 = BasicConv2d(num_filters, num_filters, kernel_size=(3, 1), padding=(1, 0))

        self.branch_1x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_1x3 = BasicConv2d(num_filters, num_filters, kernel_size=(1, 3), padding=(0, 1))

        self.branch_3x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x3 = BasicConv2d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        branch_1x3_btlneck = self.branch_1x3_btlneck(x)
        branch_3x1_btlneck = self.branch_3x1_btlneck(x)

        branch_3x1 = self.branch_3x1(branch_3x1_btlneck)
        branch_1x3 = self.branch_1x3(branch_1x3_btlneck)

        branch_3x3 = self.branch_3x3_btlneck(x)
        branch_3x3 = self.branch_3x3(branch_3x3)

        out = [branch_3x1, branch_1x3, branch_3x3]
        out = torch.cat(out, 1)

        return out


class Inception(BaseBinaryClassifier):
    def __init__(self, num_feature_planes, inner, output_features, fc1, num_classes=1, fold_number=0, gain=0.01,
                 model_prefix=""):
        positional = [num_feature_planes, inner, output_features, fc1]
        named = {"num_classes": num_classes, "fold_number": fold_number, "gain": gain}
        super().__init__(pos_params=positional, named_params=named, model_name="Inception",
                         best_model_name="./models/inception_%s_best_%s.mdl" % (model_prefix, fold_number))
        self.fold_number = fold_number
        self.activation = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception_a = _InceptionA(num_feature_planes, inner)
        self.inception_c = _InceptionC(inner * 2, inner)
        self.inception_e_1 = _InceptionE(inner * 3, inner)
        self.inception_e_2 = _InceptionE(inner * 3, inner)

        self.out_block = BasicConv2d(inner * 3, output_features, kernel_size=1)

        self.fc1 = nn.Linear(output_features * 4 * 4, fc1)
        self.fc2 = nn.Linear(fc1, num_classes)
        nn.init.xavier_normal(self.fc1.weight, gain=gain)
        nn.init.xavier_normal(self.fc2.weight, gain=gain)

    def forward(self, x):
        out = self.inception_a(x)
        out = self.max_pool(out)
        out = self.inception_c(out)
        out = self.max_pool(out)
        out = self.inception_e_1(out)
        out = self.max_pool(out)
        out = self.inception_e_2(out)
        out = self.max_pool(out)
        out = self.out_block(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    a = torch.randn(4, 5, 75, 75)
    inc = Inception(5, 32, 32, 16)
    a = Inception.to_var(a, use_gpu=False)
    res = inc.predict(a)
    print(res)
