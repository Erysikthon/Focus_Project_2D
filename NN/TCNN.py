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

import torch
import torch.nn as nn

class TCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(1, 25, (9, 9, 9), (1, 2, 2), padding=0)
        self.relu_1 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm3d(25, momentum=0.1)

        self.conv3d_2 = nn.Conv3d(25, 44, (5, 5, 5), (1, 1, 1), padding=0)
        self.relu_2 = nn.ReLU()
        self.maxpool3d_2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3d_3 = nn.Conv3d(44, 64, (3, 3, 3), (1, 1, 1), padding=0)
        self.relu_3 = nn.ReLU()
        self.batchnorm_3 = nn.BatchNorm3d(64, momentum=0.1)
        self.maxpool3d_3 = nn.MaxPool3d((1, 4, 4), (1, 2, 2))

        # We'll initialize this later dynamically based on conv output
        self.fc_4 = nn.Linear(64*21*28, 5120)
        self.relu_4 = nn.ReLU()
        self.fc_5 = nn.Linear(5120, 720)
        self.relu_5 = nn.ReLU()
        self.fc_6 = nn.Linear(720, 4)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.relu_1(x)
        x = self.batchnorm_1(x)

        x = self.conv3d_2(x)
        x = self.relu_2(x)
        x = self.maxpool3d_2(x)

        x = self.conv3d_3(x)
        x = self.relu_3(x)
        x = self.batchnorm_3(x)
        x = self.maxpool3d_3(x)

        B, C, D, H, W = x.shape

        #print(C, H, W)

        # Flatten channels + spatial dimensions for each time step
        x = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
        x = x.reshape(B, D, C*H*W)    # [B, D, C*H*W]
        
        # Apply FC to each time step
        x = self.fc_4(x)
        x = self.relu_4(x)

        x = self.fc_5(x)
        x = self.relu_5(x)

        x = self.fc_6(x)

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
