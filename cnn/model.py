import math
import torch
from torch import nn
from base.model import BaseBinaryClassifier


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.elu(out)

        return out


class ResNet(BaseBinaryClassifier):
    def __init__(self, block, num_feature_planes, layers, num_classes=1, fold_number=0):
        positional = [None, num_feature_planes, layers]
        self.fold_number = fold_number
        self.inplanes = 32
        super().__init__(pos_params=positional, named_params={}, model_name="ResNet",
                         best_model_name="./models/best_fold_%s.mdl" % fold_number)
        self.conv1 = nn.Conv2d(num_feature_planes, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 48, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 16)
        self.fc2 = nn.Linear(16, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


class LeNet(BaseBinaryClassifier):
    def __init__(self, feature_planes, conv_layers, fc1, fc2, num_classes=1, fold_number=0,
                 kernel_size=3, gain=0.01, padding=1, model_prefix=""):
        positional = [feature_planes, conv_layers, fc1, fc2]
        named = {"num_classes": num_classes, "fold_number": fold_number, "kernel_size": kernel_size,
                 "gain": gain, "padding": padding, "model_prefix": model_prefix}
        super().__init__(pos_params=positional, named_params=named, model_name="LeNet",
                         best_model_name="./models/%s_best_%s.mdl" % (model_prefix, fold_number))
        self.fold_number = fold_number
        self.activation = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        layers = []
        prev_layer = feature_planes
        for i, layer_size in enumerate(conv_layers):
            conv = nn.Conv2d(prev_layer, layer_size, kernel_size, padding=padding, bias=False)
            nn.init.xavier_normal(conv.weight, gain=gain)
            bn = nn.BatchNorm2d(layer_size, momentum=0.01)
            max_pool = nn.MaxPool2d(2, stride=2)
            layers.extend([conv, bn, self.activation, max_pool])
            prev_layer = layer_size

        self.feature_extractor = nn.Sequential(*layers)
        self.fc1 = nn.Linear(conv_layers[-1] * 4 * 4, fc1)
        self.fc2 = nn.Linear(fc1, num_classes)

        nn.init.xavier_normal(self.fc1.weight, gain=gain)
        nn.init.xavier_normal(self.fc2.weight, gain=gain)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    net = LeNet(2, (64, 128, 128, 64), 512, 256)
    opt = torch.optim.Adam(net.parameters())
    path = "/tmp/dummy.mdl"
    net.save(path, opt, False, scores={"dummy": 0.23})

    net.load(path)
    new_model = LeNet.restore(path)
    print()
