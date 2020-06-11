"""
"""

import torch
import torch.nn
import torch.optim


import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy
import pandas as pd
import numpy as np
# from word2vec import aiwaf_class
from aiwaf_dataset import aiwaf_class
from model import aiwafNet
from torch.utils.data.dataset import random_split

# 定义模型对模型进行训练此处为一个定义模型训练过程的函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in dataloaders[phase]:
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if (i+1) % 100 == 0 and phase == 'train':
                    print("\r", 'step: {}/{} '.format(i, dataset_sizes[phase]//batch_size), end='', flush=True)
                    train_loss.append(running_loss)
                    train_acc.append(running_corrects.cpu().item())
                if (i+1) % 100 == 0 and phase == 'val':
                    val_acc.append(running_corrects.cpu().item())
            print()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    for i in range(len(train_acc), -1, -1):
        if i%14 != 0:
            train_acc[i] -= train_acc[i-1]
            train_loss[i] -= train_loss[i-1]
    import json
    file = open('train_acc.txt', 'w')
    file.write(json.dumps(train_acc))
    file.close()
    file = open('train_loss.txt', 'w')
    file.write(json.dumps(train_loss))
    file.close()
    file = open('val_acc.txt', 'w')
    file.write(json.dumps(val_acc))
    file.close()

    # torch.save(model, 'model_aiwaf.pth')
    return model

if __name__ == '__main__':
    batch_size = 512
    SEQ_LEN = 100
    EMBED_DIM = 12
    NUN_CLASS = 3
    N_EPOCHS = 5
    hidden = [400, 20]

    aiwaf_datasets = aiwaf_class(seq_len=SEQ_LEN)
    train_len = int(len(aiwaf_datasets) * 0.9)
    aiwaf_datasets_train, aiwaf_datasets_val = \
        random_split(aiwaf_datasets, [train_len, len(aiwaf_datasets) - train_len])

    dataloaders = {'train': torch.utils.data.DataLoader(aiwaf_datasets_train, batch_size=batch_size,
                                                  shuffle=False, num_workers=2),
                   'val': torch.utils.data.DataLoader(aiwaf_datasets_val, batch_size=batch_size,
                                                        shuffle=False, num_workers=2)}
    dataset_sizes = {'train': len(aiwaf_datasets_train), 'val': len(aiwaf_datasets_val)}
    class_names = ['white', 'sqli', 'xss']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = aiwafNet(SEQ_LEN, EMBED_DIM, hidden, NUN_CLASS)
    model_ft = model_ft.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs使用学习率缩减
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=N_EPOCHS)


