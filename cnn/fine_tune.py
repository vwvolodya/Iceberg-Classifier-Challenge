import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
from cnn.dataset import IcebergDataset, Flip, Rotate, ToTensor, Scale
from tensorboardX import SummaryWriter

WRITER = SummaryWriter()


ONE_TRANSFORM = transforms.Compose(
        [
            Scale((224, 224)),
            Flip(axis=2, rnd=True),
            Flip(axis=1, rnd=True),
            Rotate(90, rnd=True),
            Rotate(180, rnd=True),
            ToTensor()
        ]
    )
SECOND_TRANSFORM = transforms.Compose([Scale((224, 224)), ToTensor()])


def train_one_config(num_folds):
    scores = []

    for f in range(num_folds):
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)

        model_ft = model_ft.cuda()

        criterion = nn.BCEWithLogitsLoss()

        train_set = IcebergDataset("../data/folds/train_%s.npy" % f, transform=ONE_TRANSFORM,
                                   add_feature_planes="simple")

        val_ds = IcebergDataset("../data/folds/test_%s.npy" % f, transform=SECOND_TRANSFORM,
                                add_feature_planes="simple")

        train_loader = DataLoader(train_set, batch_size=128, num_workers=6,
                                  pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, num_workers=6, pin_memory=True)
        dataloaders = {"train": train_loader, "val": val_loader}
        dataset_sizes = {"train": len(train_set), "val": len(val_ds)}

        optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.0001)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=50)
        print()
    return scores


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # model.float()
    best_model_wts = model.state_dict()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data["inputs"], data["targets"]

                # wrap them in Variable

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                preds = (outputs.data > 0.5).float()
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            WRITER.add_scalar('loss', epoch_loss, epoch)
            WRITER.add_scalar('acc', epoch_acc, epoch)
            WRITER.add_text('Text', 'text logged at step:' + str(epoch), epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Observe that all parameters are being optimized

if __name__ == "__main__":
    train_one_config(4)
