import math
from torch import nn
import numpy as np
from tqdm import tqdm as progressbar
from sklearn import metrics
from base.model import BaseModel


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


class ResNet(BaseModel):

    def __init__(self, block, num_feature_planes, layers, num_classes=1):
        self.inplanes = 32
        super().__init__()
        self.conv1 = nn.Conv2d(num_feature_planes, 32, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128 * 5 * 5, num_classes)

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
        # x = self.layer4(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    def predict(self, x, return_classes=False):
        predictions = self.__call__(x)
        classes = None
        if return_classes:
            classes = self._get_classes(predictions)
        return predictions, classes

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)
        inputs, labels = next_batch["inputs"], next_batch["targets"]
        inputs, labels = cls.to_var(inputs), cls.to_var(labels)
        return inputs, labels

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        prefix = "val_" if not training else ""
        if predictions_are_classes:
            recall = metrics.recall_score(target_y, pred_y, pos_label=1.0)
            precision = metrics.precision_score(target_y, pred_y, pos_label=1.0)
            result = {"precision": precision, "recall": recall}
        else:
            fpr, tpr, thresholds = metrics.roc_curve(target_y, pred_y, pos_label=1.0)
            auc = metrics.auc(fpr, tpr)
            result = {"auc": auc}

        final = {}
        for k, v in result.items():
            final[prefix + k] = v
        return final

    def evaluate(self, logger, loader, loss_fn=None, switch_to_eval=False):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics_1 = self._compute_metrics(self._predictions["target"], self._predictions["predicted"])
        train_metrics_2 = self._compute_metrics(self._predictions["target"], self._predictions["probs"],
                                                predictions_are_classes=False)
        train_metrics = {"train_loss": train_loss}
        train_metrics.update(train_metrics_1)
        train_metrics.update(train_metrics_2)

        if switch_to_eval:
            self.eval()
        iterator = iter(loader)
        iter_per_epoch = len(loader)
        all_predictions = np.array([])
        all_targets = np.array([])
        all_probs = np.array([])
        losses = []
        for i in range(iter_per_epoch):
            inputs, targets = self._get_inputs(iterator)
            probs, classes = self.predict(inputs, return_classes=True)
            target_y = self.to_np(targets).squeeze()
            if loss_fn:
                loss = loss_fn(probs, targets)
                losses.append(loss.data[0])
            probs = self.to_np(probs).squeeze()
            all_targets = np.append(all_targets, target_y)
            all_probs = np.append(all_probs, probs)
            all_predictions = np.append(all_predictions, classes)
        computed_metrics = self._compute_metrics(all_targets, all_predictions, training=False)
        computed_metrics_1 = self._compute_metrics(all_targets, all_probs, training=False,
                                                   predictions_are_classes=False)

        val_loss = sum(losses) / len(losses)
        computed_metrics.update({"val_loss": val_loss})
        computed_metrics.update(computed_metrics_1)
        if switch_to_eval:
            # switch back to train
            self.train()

        self._log_and_reset(logger, data=train_metrics, log_grads=True)
        self._log_and_reset(logger, data=computed_metrics, log_grads=False)

        self._reset_predictions_cache()
        return computed_metrics

    def fit(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        for e in progressbar(range(num_epochs)):
            iter_per_epoch = len(data_loader)
            self._epoch = e
            data_iter = iter(data_loader)
            for i in range(iter_per_epoch):
                inputs, labels = self._get_inputs(data_iter)

                predictions, classes = self.predict(inputs, return_classes=True)

                optim.zero_grad()
                loss = loss_fn(predictions, labels)
                loss.backward()
                optim.step()

                self._accumulate_results(self.to_np(labels).squeeze(),
                                         classes,
                                         loss=loss.data[0],
                                         probs=self.to_np(predictions).squeeze())
            self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            self.save("./models/clf_%s.mdl" % str(e + 1))


if __name__ == "__main__":
    import torch
    from base.logger import Logger
    from cnn.dataset import IcebergDataset, ToTensor
    from torch.utils.data import DataLoader

    top = None
    val_top = None
    train_batch_size = 256
    test_batch_size = 64

    main_logger = Logger("../logs")

    train_ds = IcebergDataset("../data/folds/train_0.npy", transform=ToTensor(), top=top)
    val_ds = IcebergDataset("../data/folds/test_0.npy", transform=ToTensor(), top=top)

    net = ResNet(BasicBlock, 2, [2, 2, 2, 2], num_classes=1)
    net.show_env_info()
    loss_func = nn.BCELoss()

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=6, pin_memory=True)

    if torch.cuda.is_available():
        net.cuda()
        loss_func.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=0.0002)
    net.fit(optim, loss_func, train_loader, val_loader, 150, logger=main_logger)
