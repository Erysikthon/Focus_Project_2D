import torch
import numpy as np
import pandas as pd
import os
from utilities import terminal_colors
from NN.OLD_videodataset import VideoDataset
from sklearn.preprocessing import StandardScaler
import random as rd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import math
from graphs import kernel_heatmap
from graphs import loss_over_epochs_lineplot
from graphs import plot_confusion_matrix

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
    print(terminal_colors.GREEN + "\nMPS device found\n" + terminal_colors.ENDC)
else:
    print (terminal_colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + terminal_colors.ENDC)

video_names = []
for i in range(1,22):
    if not i==5:
        video_names.append(f"OFT_left_{i}.csv")

video_path = "data"
video_names_training, video_names_test = train_test_split(video_names, test_size = 4, shuffle = True)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}
points =   ['mouse_top.mouse_top_0.nose.x', 'mouse_top.mouse_top_0.nose.y',
            'mouse_top.mouse_top_0.headcentre.x','mouse_top.mouse_top_0.headcentre.y',
            'mouse_top.mouse_top_0.neck.x','mouse_top.mouse_top_0.neck.y', 
            'mouse_top.mouse_top_0.earl.x', 'mouse_top.mouse_top_0.earl.y',
            'mouse_top.mouse_top_0.earr.x','mouse_top.mouse_top_0.earr.y', 
            'mouse_top.mouse_top_0.bodycentre.x','mouse_top.mouse_top_0.bodycentre.y',
            'mouse_top.mouse_top_0.bcl.x','mouse_top.mouse_top_0.bcl.y',
            'mouse_top.mouse_top_0.bcr.x', 'mouse_top.mouse_top_0.bcr.y',
            'mouse_top.mouse_top_0.hipl.x','mouse_top.mouse_top_0.hipl.y', 
            'mouse_top.mouse_top_0.hipr.x', 'mouse_top.mouse_top_0.hipr.y',
            'mouse_top.mouse_top_0.tailbase.x','mouse_top.mouse_top_0.tailbase.y',
            'mouse_top.mouse_top_0.tailcentre.x','mouse_top.mouse_top_0.tailcentre.y',
            'mouse_top.mouse_top_0.tailtip.x', 'mouse_top.mouse_top_0.tailtip.y',
            'oft_3d.oft_3d_0.tl.x', 'oft_3d.oft_3d_0.tl.y',
            'oft_3d.oft_3d_0.tr.x','oft_3d.oft_3d_0.tr.y', 
            'oft_3d.oft_3d_0.bl.x', 'oft_3d.oft_3d_0.bl.y',
            'oft_3d.oft_3d_0.br.x','oft_3d.oft_3d_0.br.y',
            'oft_3d.oft_3d_0.top_tl.x', 'oft_3d.oft_3d_0.top_tl.y',
            'oft_3d.oft_3d_0.top_tr.x','oft_3d.oft_3d_0.top_tr.y',
            'oft_3d.oft_3d_0.top_bl.x', 'oft_3d.oft_3d_0.top_bl.y',
            'oft_3d.oft_3d_0.top_br.x','oft_3d.oft_3d_0.top_br.y']

scaler = StandardScaler().set_output(transform = "pandas")
for name in video_names_training:
    video = pd.read_csv(video_path+"/features/"+name)
    scaler.partial_fit(video)
print(terminal_colors.GREEN + "Partial fitted: " + terminal_colors.ENDC + f"{scaler}\n")

train_set = VideoDataset(path = video_path, 
                          video_names = video_names_training,
                          scaler = scaler,
                          frames = 18000,
                          behaviors = behaviors,
                          points = points,
                          identity = "train dataset")

test_set = VideoDataset(path = video_path, 
                          video_names = video_names_test,
                          scaler = scaler,
                          frames = 18000,
                          behaviors = behaviors,
                          points = points,
                          identity = "test dataset")

train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_data_loader = DataLoader(test_set, batch_size=1, shuffle=False)

class Block(nn.Module):
    def __init__(self, features_in : int, features_out : int, kernel_size : int, dropout: int):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(features_in, features_out, kernel_size=kernel_size, padding = (kernel_size-1)//2)
        self.batch1d_1 = nn.BatchNorm1d(features_out)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout)
    
    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.batch1d_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        return x

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = Block(42, 256, 3, 0.05)
        self.block_2 = Block(256, 256, 5, 0.05)
        self.block_3 = Block(256, 512, 3, 0.15)
        self.block_4 = Block(512, 1024, 3, 0.2)

        self.head = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 4, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = self.head(x)
        
        return x

network = NeuralNet().to(mps_device)

print(terminal_colors.GREEN + "Network initalized: " + terminal_colors.ENDC + f"{network}\n")

def train_loop(dataloader : DataLoader, network : NeuralNet, loss_fn : nn.CrossEntropyLoss, optimizer : torch.optim.RMSprop):

    total_loss = 0
    y_true = []
    y_pred = []

    network.train()
    with tqdm(desc = terminal_colors.CYAN +"    train" + terminal_colors.ENDC, total = len(dataloader), ascii = True) as pbar:
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(mps_device), y.to(mps_device)
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
    print(terminal_colors.WARNING + f"        loss value:" + terminal_colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()
    return mean_loss, y_true, y_pred

def test_loop(dataloader : DataLoader, network : NeuralNet, loss_fn : nn.CrossEntropyLoss,):
    network.eval()
    total_loss = 0
    y_true = []
    y_pred = []


    with torch.no_grad():
        with tqdm(desc = terminal_colors.CYAN +"    test" + terminal_colors.ENDC, total = len(dataloader), ascii = True) as pbar:
            for X, y in dataloader:
                X, y = X.to(mps_device), y.to(mps_device)
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
    print(terminal_colors.WARNING + f"        loss value:" + terminal_colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()
    return mean_loss, y_true, y_pred

class_weights = torch.tensor([1.0, 4, 10, 10]).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.RMSprop(network.parameters(), lr = 1e-4)
test_total_loss = []
train_total_loss = []

for epoch in range(1,101):

    print(terminal_colors.GREEN + f"\nEpoch:" + terminal_colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer)
    test_mean_loss, y_true_test, y_pred_test = test_loop(test_data_loader, network, loss_function)
    test_total_loss.append(test_mean_loss)
    train_total_loss.append(train_mean_loss)
    
    if epoch % 20 == 0:
        print(terminal_colors.WARNING + f"\nclassification report epoch: " + terminal_colors.ENDC + f" {epoch}")

        print(terminal_colors.CYAN + f"    train: " + terminal_colors.ENDC)
        print(classification_report(
            y_true_train,
            y_pred_train,
            target_names=behaviors.keys()
        ))
        plot_confusion_matrix(y_true_train, y_pred_train, behaviors, f"./output/train_confusion_matrix_at_{epoch}.png")

        print(terminal_colors.CYAN + f"\n    test: " + terminal_colors.ENDC)
        print(classification_report(
            y_true_test,
            y_pred_test,
            target_names=behaviors.keys()
        ))
        plot_confusion_matrix(y_true_test, y_pred_test, behaviors, f"./output/test_confusion_matrix_at_{epoch}.png")

        pd.DataFrame(y_pred_test).to_csv(f"./output/y_pred_{epoch}.csv")
        pd.DataFrame(y_true_test).to_csv(f"./output/y_true_{epoch}.csv")

        kernel_heatmap(network.block_1.conv1d_1, f"./output/conv1d_1_kernels_epoch_{epoch}.png", n_kernels = 18)           


loss_over_epochs_lineplot(train_total_loss, f"./output/train_loss_vs_{len(train_total_loss)}_epochs.png")
loss_over_epochs_lineplot(test_total_loss, f"./output/test_loss_vs_{len(test_total_loss)}_epochs.png")

