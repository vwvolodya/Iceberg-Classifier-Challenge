import abc
import cv2
import sys
import numpy as np
import torch
import random
import torch.nn as nn
import shutil
from sklearn import metrics
from tqdm import tqdm as progressbar
from torch.autograd import Variable
from collections import defaultdict
from pprint import pformat
from base.exceptions import ProjectException


class BaseModel(nn.Module):
    def __init__(self, pos_params, named_params, seed=10101, model_name=None, best_model_name="./models/best.mdl"):
        self._best_model_name = best_model_name
        self.model_name = model_name
        self._predictions = defaultdict(list)
        self._epoch = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        super().__init__()
        self._model_params = {"args": pos_params, "kwargs": named_params}

    def _reset_predictions_cache(self):
        self._predictions = defaultdict(list)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_inputs(cls, iterator):
        pass

    @abc.abstractmethod
    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        pass

    @abc.abstractmethod
    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        pass

    @classmethod
    def to_np(cls, x):
        if x is None:
            return np.array([])
        # convert Variable to numpy array
        if isinstance(x, Variable):
            return x.data.cpu().numpy()
        else:
            return x.numpy()

    @classmethod
    def to_var(cls, x, use_gpu=True, inference_only=False):
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda(async=True)
        return Variable(x, volatile=inference_only)

    @classmethod
    def to_tensor(cls, x):
        # noinspection PyUnresolvedReferences
        tensor = torch.from_numpy(x).float()
        return tensor

    def _accumulate_results(self, target_y, pred_y, loss=None, **kwargs):
        if loss is not None:
            self._predictions["train_loss"].append(loss)
        for k, v in kwargs.items():
            self._predictions[k].extend(v)
        if target_y is not None:
            self._predictions["target"].extend(target_y)
        if pred_y is not None:
            self._predictions["predicted"].extend(pred_y)

    @classmethod
    def show_env_info(cls):
        print('Python VERSION:', sys.version)
        print('CUDA VERSION')
        # noinspection PyUnresolvedReferences
        print('CUDNN VERSION:', torch.backends.cudnn.version())
        print('Number CUDA Devices:', torch.cuda.device_count())
        print('Devices')
        print("OS: ", sys.platform)
        print("PyTorch: ", torch.__version__)
        print("Numpy: ", np.__version__)
        use_cuda = torch.cuda.is_available()
        print("CUDA is available", use_cuda)

    def summary(self):
        print("----------==================------------")
        print(repr(self))
        print("----------==================------------")

    def _log_data(self, logger, data_dict):
        for tag, value in data_dict.items():
            logger.scalar_summary(tag, value, self._epoch + 1)

    def _log_grads(self, logger):
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, self.to_np(value), self._epoch + 1)
            if value.grad is not None:
                logger.histo_summary(tag + '/grad', self.to_np(value.grad), self._epoch + 1)

    def _log_and_reset(self, logger, data, log_grads=True):
        self._log_data(logger, data)
        if log_grads:
            self._log_grads(logger)

    def save(self, path, optimizer, is_best, scores=None):
        # positional and named params will be used to restore model later
        data = {
            'epoch': self._epoch + 1,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_params': self._model_params,
            'scores': scores
        }
        torch.save(data, path)
        if is_best:
            shutil.copyfile(path, self._best_model_name)

    def load(self, path):
        """
        Load model state from file. Model must be initialised by this moment
        :param path: string path to file containing model
        :return: instance of this class
        :rtype: BaseModel
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        scores = pformat(checkpoint["scores"])
        print("Loading model from epoch %s with scores \n%s" % (checkpoint["epoch"], scores))
        self.eval()
        return self

    @classmethod
    def restore(cls, path):
        """
        Create instance of class and load parameters
        :param path: string path to model file
        :return: instance of this class
        :rtype: BaseModel
        """
        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]
        scores = pformat(checkpoint["scores"])
        model_params = checkpoint["model_params"]
        if model_params is not None:
            positional = model_params["args"]
            named = model_params["kwargs"]
        else:
            raise ProjectException("No model params found. Cannot create instance of class!")
        print("Going to restore model from params:\n%s %s from epoch %s with scores \n"
              "%s" %(positional, named, epoch, scores))
        instance = cls(*positional, **named)
        instance.load_state_dict(checkpoint['state_dict'])
        instance.eval()
        return instance


class BaseBinaryClassifier(BaseModel):
    @classmethod
    def _get_classes(cls, predictions):
        classes = (predictions.data > 0.5).float()
        pred_y = classes.cpu().numpy().squeeze()
        return pred_y

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

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
            accuracy = metrics.accuracy_score(target_y, pred_y)
            result = {"precision": precision, "recall": recall, "acc": accuracy}
        else:
            fpr, tpr, thresholds = metrics.roc_curve(target_y, pred_y, pos_label=1.0)
            auc = metrics.auc(fpr, tpr)
            result = {"auc": auc}

        final = {}
        for k, v in result.items():
            final[prefix + k] = v
        return final

    def _eval_on_validation(self, loader, loss_fn):
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
        return computed_metrics

    def evaluate(self, logger, loader, loss_fn=None, switch_to_eval=True):
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
        computed_metrics = self._eval_on_validation(loader, loss_fn)
        if switch_to_eval:
            # switch back to train
            self.train()

        self._log_and_reset(logger, data=train_metrics, log_grads=True)
        self._log_and_reset(logger, data=computed_metrics, log_grads=False)

        self._reset_predictions_cache()
        return computed_metrics

    def fit(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        best_loss = float("inf")
        for e in progressbar(range(num_epochs)):
            self._epoch = e + 1

            iter_per_epoch = len(data_loader)
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
            stats = self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            is_best = stats["val_loss"] < best_loss
            best_loss = min(best_loss, stats["val_loss"])
            self.save("./models/%s_%s_fold_%s.mdl" % (self.model_name, str(e + 1), self.fold_number),
                      optim, is_best, scores=stats)
        return best_loss


class BaseAutoEncoder(BaseModel):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        pass

    def predict(self, x, **kwargs):
        predictions = self.__call__(x)
        return predictions

    @classmethod
    def _get_inputs(cls, iterator):
        next_batch = next(iterator)
        inputs, targets = next_batch["inputs"], next_batch["targets"]
        inputs, targets = cls.to_var(inputs), cls.to_var(targets)
        return inputs, targets

    def _eval_on_validation(self, loader, loss_fn, logger):
        iterator = iter(loader)
        iter_per_epoch = len(loader)

        losses = []
        inputs, targets, probs = None, None, None
        for i in range(iter_per_epoch):
            inputs, targets = self._get_inputs(iterator)
            probs = self.predict(inputs)
            if loss_fn:
                loss = loss_fn(probs, targets)
                losses.append(loss.data[0])
        self._log_images(inputs, targets, probs, logger, prefix="val_")
        val_loss = sum(losses) / len(losses)
        computed_metrics = {"val_loss": val_loss, "val_loss_sqrt": val_loss ** 0.5}
        return computed_metrics

    def evaluate(self, logger, loader, loss_fn=None, switch_to_eval=True):
        # aggregate results from training epoch.
        train_losses = self._predictions.pop("train_loss")
        train_loss = sum(train_losses) / len(train_losses)
        train_metrics = {"train_loss": train_loss, "train_loss_sqrt": train_loss ** 0.5}

        if switch_to_eval:
            self.eval()
        computed_metrics = self._eval_on_validation(loader, loss_fn, logger)
        if switch_to_eval:
            # switch back to train
            self.train()

        self._log_and_reset(logger, data=train_metrics, log_grads=True)
        self._log_and_reset(logger, data=computed_metrics, log_grads=False)

        self._reset_predictions_cache()
        return computed_metrics

    @classmethod
    def _get_heatmaps(cls, images):
        images = [cv2.normalize(i.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) for i in images]
        images = [np.round(i * 255).astype(np.uint8) for i in images]
        images = [cv2.applyColorMap(i, cv2.COLORMAP_BONE) for i in images]
        return images

    def _log_images(self, inputs, targets, predictions, logger, start=0, top=3, prefix=""):
        if all((inputs is None, targets is None, predictions is None)):
            return
        inputs = self.to_np(inputs)[start: start + top, :, :, :].mean(axis=1)
        targets = self.to_np(targets)[start: start + top, :, :, :].mean(axis=1)
        predictions = self.to_np(predictions)[start: start + top, :, :, :].mean(axis=1)
        inputs = self._get_heatmaps(inputs)
        targets = self._get_heatmaps(targets)
        predictions = self._get_heatmaps(predictions)
        images = {prefix + "inputs": inputs,
                  prefix + "targets": targets,
                  prefix + "predictions": predictions}
        for k, v in images.items():
            logger.image_summary(k, v, self._epoch + 1)

    def fit(self, optim, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        best_loss = float("inf")
        for e in progressbar(range(num_epochs)):
            self._epoch = e + 1
            iter_per_epoch = len(data_loader)
            data_iter = iter(data_loader)
            inputs, targets, predictions, start_point = None, None, None, random.randint(0, 10)
            for i in range(iter_per_epoch):
                inputs, targets = self._get_inputs(data_iter)

                predictions = self.predict(inputs)

                optim.zero_grad()
                loss = loss_fn(predictions, targets)
                loss.backward()
                optim.step()

                self._accumulate_results(None, None, loss=loss.data[0])
            self._log_images(inputs, targets, predictions, logger, start=start_point, prefix="train_")
            stats = self.evaluate(logger, validation_data_loader, loss_fn, switch_to_eval=True)
            is_best = stats["val_loss"] < best_loss
            best_loss = min(best_loss, stats["val_loss"])
            self.save("./models/%s_%s_fold_%s.mdl" % (self.model_name, str(e + 1), self.fold_number),
                      optim, is_best, scores=stats)
        return best_loss
