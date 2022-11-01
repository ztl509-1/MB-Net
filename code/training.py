import torch
import torch.nn as nn
import os
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from model import MG_Net
from DataLoader import FallDataset
from random_seed import setup_seed
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

path = './dataset/CW_rawdata/'
val_path = '../dataset/CW_rawdata_leave_one_person/'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
decay = 4e-5 * 10

def training(model, train_dataset, test_dataset, epochs=10, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=0.00001)

    best_loss = 100
    bce_loss = nn.BCELoss()
    train_loss_ls = []
    test_loss_ls = []

    for epoch in range(epochs):
        model.train()

        i = 0
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_dataset:
            data, labels = batch
            data = data.cuda()
            labels = labels.cuda()

            _, _, _,  preds = model(data)
            preds = preds.squeeze(1)

            loss = bce_loss(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds[preds > 0.5] = 1
            preds[~(preds > 0.5)] = 0
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += len(labels)
            i += 1

        train_acc = total_correct / total_samples
        train_loss = total_loss / i
        train_loss_ls.append(train_loss)

        print("epoch: ", epoch, "train_acc: ", train_acc, "total_loss: ", train_loss)

        test_loss = tsting(model, test_dataset)
        test_loss_ls.append(test_loss)

        lr_scheduler.step(int(train_loss*100))

        # save the model
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = model

    plt.plot(train_loss_ls)
    plt.plot(test_loss_ls)
    plt.legend(["train_loss", "test_loss"])
    plt.show()

    return best_model.cuda()


def tsting(model, test_dataset):
    model.eval()

    bce_loss = nn.BCELoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    i = 0
    for batch in test_dataset:
        data, labels = batch
        data = data.cuda()
        labels = labels.cuda()

        _, _, _, preds = model(data)

        # binary classification
        preds = preds.squeeze(1)
        loss = bce_loss(preds, labels)

        preds[preds > 0.5] = 1
        preds[~(preds > 0.5)] = 0
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += len(labels)
        i += 1

    total_loss = total_loss / i
    print("test_acc: ", total_correct / total_samples, "loss: ", total_loss)

    return total_loss


def validating(model, valid_dataset):
    model.eval()

    bce_loss = nn.BCELoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    i = 0
    for batch in valid_dataset:
        data, labels = batch
        data = data.cuda()
        labels = labels.cuda()

        _,_,_,preds = model(data)

        # binary classification
        preds = preds.squeeze(1)

        preds_ = preds.clone() #deep copy

        loss = bce_loss(preds, labels)

        preds[preds > 0.5] = 1
        preds[~(preds > 0.5)] = 0
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += len(labels)
        i += 1

    total_loss = total_loss / i
    print("test_acc: ", total_correct / total_samples, "loss: ", total_loss)
    print(roc_auc_score(labels.cpu().detach().numpy(), preds_.cpu().detach().numpy()))

    class_names = ['fall', 'non-fall']
    print(classification_report(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), target_names=class_names,
                                digits=4))

    return total_loss

if __name__ == "__main__":
    setup_seed(2)
    # step1: data reading
    dataset = FallDataset(path)
    valid_dataset = FallDataset(val_path)

    print("dataset_len: ", len(dataset))
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset_split, test_dataset_split = torch.utils.data.random_split(
        dataset,
        [train_size, test_size]
    )
    print("The size of training dataset:  ", len(train_dataset_split))

    # put into the Dataloader
    batch_size = 64
    train_dataset = torch.utils.data.DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset_split, batch_size=batch_size, shuffle=False)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    print("build the model!")
    model = MG_Net().cuda()
    model = training(model, train_dataset, test_dataset, epochs=150, learning_rate=0.002)
    validating(model, valid_dataset)




