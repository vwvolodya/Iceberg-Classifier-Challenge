import torch
import torch.nn.functional as F
from torch import nn
from base.model import BaseBinaryClassifier


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class _InceptionA(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_1x1 = BasicConv2d(features, num_filters, kernel_size=1)

        self.branch_3x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x3 = BasicConv2d(num_filters, num_filters, kernel_size=3, padding=1)

        self.branch_5x5_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_5x5 = BasicConv2d(num_filters, num_filters, kernel_size=5, padding=2)
        self.branch_pool = BasicConv2d(features, num_filters, kernel_size=1)

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3_btlneck = self.branch_3x3_btlneck(x)
        branch_5x5_btlneck = self.branch_5x5_btlneck(x)

        branch_3x3 = self.branch_3x3(branch_3x3_btlneck)
        branch_5x5 = self.branch_5x5(branch_5x5_btlneck)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        out = [branch_1x1, branch_3x3, branch_5x5, branch_pool]
        out = torch.cat(out, 1)
        return out


class _InceptionB(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_3x3 = BasicConv2d(features, num_filters, kernel_size=3, stride=2)

    def forward(self, x):
        branch_3x3 = self.branch_3x3(x)
        max_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        out = [branch_3x3, max_pool]
        out = torch.cat(out, 1)
        return out


class _InceptionC(nn.Module):
    def __init__(self, features, num_filters):
        super().__init__()
        self.branch_1x1 = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_5x1_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_1x5_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_pool = BasicConv2d(features, num_filters, kernel_size=1)

        self.branch_5x1 = BasicConv2d(num_filters, num_filters, kernel_size=(5, 1), padding=(2, 0))
        self.branch_1x5 = BasicConv2d(num_filters, num_filters, kernel_size=(1, 5), padding=(0, 2))

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)

        branch_5x1_btlneck = self.branch_5x1_btlneck(x)
        branch_5x1 = self.branch_5x1(branch_5x1_btlneck)

        branch_1x5_btlneck = self.branch_1x5_btlneck(x)
        branch_1x5 = self.branch_1x5(branch_1x5_btlneck)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch_1x1, branch_5x1, branch_1x5, branch_pool]
        out = torch.cat(out, 1)
        return out


class _InceptionE(nn.Module):
    def __init__(self, features, num_filters, output_features):
        super().__init__()
        self.branch_1x1 = BasicConv2d(features, num_filters, kernel_size=1)

        self.branch_3x1_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_3x1 = BasicConv2d(num_filters, num_filters, kernel_size=(3, 1), padding=(1, 0))

        self.branch_1x3_btlneck = BasicConv2d(features, num_filters, kernel_size=1)
        self.branch_1x3 = BasicConv2d(num_filters, num_filters, kernel_size=(1, 3), padding=(0, 1))

        self.branch_pool = BasicConv2d(features, num_filters, kernel_size=1)
        self.out_block = BasicConv2d(num_filters * 4, output_features, kernel_size=1)

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_1x3_btlneck = self.branch_1x3_btlneck(x)
        branch_3x1_btlneck = self.branch_3x1_btlneck(x)

        branch_3x1 = self.branch_3x1(branch_3x1_btlneck)
        branch_1x3 = self.branch_1x3(branch_1x3_btlneck)

        pool_branch = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        pool_branch = self.branch_pool(pool_branch)

        out = [branch_1x1, branch_3x1, branch_1x3, pool_branch]
        out = torch.cat(out, 1)

        out = self.out_block(out)
        return out


class Inception(BaseBinaryClassifier):
    def __init__(self, num_feature_planes, conv, fc, fc2, num_classes=1, fold_number=0, gain=0.01, model_prefix=""):
        super().__init__(best_model_name="./models/inception_%s_best_%s.mdl" % (model_prefix, fold_number))
        self.fold_number = fold_number
        self.activation = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_1 = BasicConv2d(num_feature_planes, conv, kernel_size=3, padding=1)

        inner = 24
        self.inception_a = _InceptionA(conv, inner)
        self.inception_b = _InceptionB(inner * 4, inner)
        self.inception_c = _InceptionC(inner * 4 + inner, inner)
        self.inception_e = _InceptionE(inner * 4, inner, output_features=16)

        self.fc = nn.Linear(16 * 9 * 9, num_classes)
        nn.init.xavier_normal(self.fc.weight, gain=gain)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.inception_a(out)
        out = self.max_pool(out)
        out = self.inception_b(out)
        out = self.inception_c(out)
        out = self.inception_e(out)
        out = self.max_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    a = torch.randn(4, 5, 75, 75)
    inc = Inception(5, 48, None, None)
    a = Inception.to_var(a, use_gpu=False)
    res = inc.predict(a)
    print(res)
