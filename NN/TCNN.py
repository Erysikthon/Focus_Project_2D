import torch
import numpy as np
import pandas as pd
import os
from utilities import terminal_colors as colors
from VideoDataSet import RandomizedDataset, SingleVideoDataset, SingleVideoDatasetCollection
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn
from sklearn.metrics import classification_report
from graphs import kernel_heatmap_3d
from graphs import loss_over_epochs_lineplot
from graphs import plot_confusion_matrix

printo = False

import torch
import torch.nn as nn

class TCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(1, 30, (5, 5, 5), (1, 2, 2), padding=0)
        self.bn1 = nn.BatchNorm3d(30)
        self.relu_1 = nn.ReLU()

        self.conv3d_2 = nn.Conv3d(30, 80, (5, 5, 5), (1, 1, 1), padding=0)
        self.bn2 = nn.BatchNorm3d(80)
        self.relu_2 = nn.ReLU()
        self.maxpool3d_2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3d_3 = nn.Conv3d(80, 140, (5, 3, 3), (1, 1, 1), padding=0)
        self.bn3 = nn.BatchNorm3d(140)
        self.relu_3 = nn.ReLU()
        self.maxpool3d_3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.fc_4 = nn.Linear(140*15*7, 4800)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout(0.3)

        self.fc_5 = nn.Linear(4800, 5)

    def forward(self, x):

        if printo:
            B, C, D, H, W = x.shape
            print(C, H, W)

        x = self.conv3d_1(x)
        x = self.bn1(x)
        x = self.relu_1(x)

        if printo:
            B, C, D, H, W = x.shape
            print(C, H, W)

        x = self.conv3d_2(x)
        x = self.bn2(x)
        x = self.relu_2(x)
        x = self.maxpool3d_2(x)

        if printo:
            B, C, D, H, W = x.shape
            print(C, H, W)

        x = self.conv3d_3(x)
        x = self.bn3(x)
        x = self.relu_3(x)
        x = self.maxpool3d_3(x)

        B, C, D, H, W = x.shape
        if printo:
            print(C, H, W)

        x = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
        x = x.reshape(B, D, C*H*W)    # [B, D, C*H*W]
        
        x = self.fc_4(x)
        x = self.relu_4(x)
        x = self.dropout_4(x)

        x = self.fc_5(x)

        x = x.permute(0, 2, 1)

        return x


def train_loop(dataloader : DataLoader, network : TCNN, loss_fn : nn.CrossEntropyLoss, optimizer : torch.optim.RMSprop, device: torch.device):

    total_loss = 0
    y_true = []
    y_pred = []

    network.train()
    with tqdm(desc = colors.CYAN +"    train" + colors.ENDC, total = len(dataloader), ascii = True) as pbar:
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y = y.long()
            optimizer.zero_grad()

            pred = network(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

            pred = pred.transpose(1,2)
            pred = pred.argmax(2)

            y_true.append(y.detach().cpu())
            y_pred.append(pred.detach().cpu())

            pbar.update(1)

    mean_loss = total_loss/len(dataloader)
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()
    return mean_loss, y_true, y_pred

def test_loop(dataloader : DataLoader, network : TCNN, loss_fn : nn.CrossEntropyLoss, device: torch.device):
    network.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        with tqdm(desc = colors.CYAN +"    test" + colors.ENDC, total = len(dataloader), ascii = True) as pbar:
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y = y.long()

                pred = network(X)
                loss = loss_fn(pred, y)

                total_loss += loss.detach().cpu().item()

                pred = pred.transpose(1,2)
                pred = pred.argmax(2)

                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

                pbar.update(1)
        
    mean_loss = total_loss/len(dataloader)
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()

    return mean_loss, y_true, y_pred
