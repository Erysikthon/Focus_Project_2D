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

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.H = nn.modules.Sequential(
        nn.modules.Conv3d(channels, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        nn.modules.BatchNorm3d(channels),
        nn.modules.ReLU(),
        nn.modules.Conv3d(channels, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        nn.modules.BatchNorm3d(channels)
        )

    def forward(self, x):
        x = self.H(x)
        return x

class TCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_convolution = nn.modules.Sequential(
            nn.modules.Conv3d(1, 32, (5, 5, 5), (1, 2, 2), padding = 0),
            nn.modules.BatchNorm3d(32),
            nn.modules.ReLU(),
            nn.modules.Conv3d(32, 48, (5, 5, 5), (1, 1, 1), padding = 0),
            nn.modules.BatchNorm3d(48),
            nn.modules.ReLU(),
            nn.modules.MaxPool3d((3, 2, 2), (2, 2, 2))
        )

        self.res_population_1 = nn.ModuleList()
        for i in range(0, 10):
            self.res_population_1.append(ResBlock(48))

        self.switch_1_2 = nn.modules.Sequential(
            nn.modules.Conv3d(48, 64, (5, 5, 5), (1, 1, 1), padding = 0),
            nn.modules.BatchNorm3d(64),
            nn.modules.ReLU(),
            nn.modules.MaxPool3d((3, 2, 2), (2, 2, 2))
        )
        
        self.res_population_2 = nn.ModuleList()
        for i in range(0, 10):
            self.res_population_2.append(ResBlock(64))
        
        self.final_convolution = nn.modules.Sequential(
            nn.modules.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), padding = 0),
            nn.modules.BatchNorm3d(64),
            nn.modules.ReLU(),
            nn.modules.Conv3d(64, 64, (3, 3, 3), (1, 1, 1), padding = 0),
            nn.modules.BatchNorm3d(64)
        )

        self.fc_4 = nn.Linear(64*10*2, 1000)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout(0.5)

        self.fc_5 = nn.Linear(1000, 100)
        self.relu_5 = nn.ReLU()
        self.dropout_5 = nn.Dropout(0.3)

        self.fc_6 = nn.Linear(100, 5)

    def forward(self, x : torch.Tensor):

        x = self.initial_convolution(x)

        if printo:
            B, C, D, H, W = x.shape
            print(f"{C}*{H}*{W} - {D}")
        
        for res_block in self.res_population_1:
            x = x + res_block(x)

        x = self.switch_1_2(x)

        if printo:
            B, C, D, H, W = x.shape
            print(f"{C}*{H}*{W} - {D}")

        for res_block in self.res_population_2:
            x = x + res_block(x)
        
        x = self.final_convolution(x)

        B, C, D, H, W = x.shape
        if printo:
            print(f"{C}*{H}*{W} - {D}")

        x = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
        x = x.reshape(B, D, C*H*W)    # [B, D, C*H*W]
        
        x = self.fc_4(x)
        x = self.relu_4(x)
        x = self.dropout_4(x)

        x = self.fc_5(x)
        x = self.relu_5(x)
        x = self.dropout_5(x)

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
