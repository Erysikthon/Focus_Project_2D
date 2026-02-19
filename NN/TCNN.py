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


r = 19

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device = mps_device)
    print(colors.GREEN + "\nMPS device found\n" + colors.ENDC)
else:
    print (colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + colors.ENDC)

video_names = []
for i in range(1,22):
    if not i==5:
        video_names.append(f"OFT_left_{i}")

features_folder = "./data/rotated_videos"
labels_folder = "./data/labels"
video_names_train, video_names_test = train_test_split(video_names, test_size = 1, shuffle = True, random_state = 42)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, 98, r, 6, random_state = 42, identity = "TRAIN randomized dataset")
test_set = SingleVideoDatasetCollection(features_folder, labels_folder, video_names_test, behaviors, 98, r, identity = "TEST single dataset collection")

train_data_loader = DataLoader(train_set, 2)
test_data_loader = DataLoader(test_set, 2)

"""
train_set.__getitem__(0, debug = True)
test_set_collection[0].__getitem__(0, debug = True)
test_set_collection[0].__getitem__(1, debug = True)
"""

class TCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 48, (7, 7, 7), (1, 2, 2), padding = 0)
        self.relu_1 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm3d(48, momentum=0.001)

        self.conv3d_2 = nn.Conv3d(48, 124, (5, 5, 5), (1, 1, 1), padding = 0)
        self.relu_2 = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm3d(124, momentum=0.001)

        self.conv3d_3 = nn.Conv3d(124, 256, (5, 5, 5), (1, 1, 1), padding = 0)
        self.relu_3 = nn.ReLU()
        self.batchnorm_3 = nn.BatchNorm3d(256, momentum=0.001)
        self.maxpool3d_3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3d_4 = nn.Conv3d(256, 142, (5, 5, 5), (1, 1, 1), padding = 0)
        self.relu_4 = nn.ReLU()
        self.batchnorm_4 = nn.BatchNorm3d(142, momentum=0.001)

        self.conv1d_5 = nn.Conv1d(142, 4, kernel_size=1)

    def forward(self, x):

        x = self.conv3d_1(x)
        x = self.relu_1(x)
        x = self.batchnorm_1(x)

        x = self.conv3d_2(x)
        x = self.relu_2(x)
        x = self.batchnorm_2(x)

        x = self.conv3d_3(x)
        x = self.relu_3(x)
        x = self.batchnorm_3(x)
        x = self.maxpool3d_3(x)

        x = self.conv3d_4(x)
        x = self.relu_4(x)
        x = self.batchnorm_4(x)

        x = x.mean(dim=(-1, -2))
        x = self.conv1d_5(x)
        
        return x


network = TCNN().to(mps_device)
print(colors.GREEN + "Network initalized: " + colors.ENDC + f"{network}\n")

def train_loop(dataloader : DataLoader, network : TCNN, loss_fn : nn.CrossEntropyLoss, optimizer : torch.optim.RMSprop):

    total_loss = 0
    y_true = []
    y_pred = []

    network.train()
    with tqdm(desc = colors.CYAN +"    train" + colors.ENDC, total = len(dataloader), ascii = True) as pbar:
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
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()
    return mean_loss, y_true, y_pred

def test_loop(dataloader : DataLoader, network : TCNN, loss_fn : nn.CrossEntropyLoss):
    network.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        with tqdm(desc = colors.CYAN +"    test" + colors.ENDC, total = len(dataloader), ascii = True) as pbar:
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
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {mean_loss}")

    y_true = torch.cat(y_true).numpy().flatten()
    y_pred = torch.cat(y_pred).numpy().flatten()

    return mean_loss, y_true, y_pred

class_weights = torch.tensor([1.0, 4, 10, 10]).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.RMSprop(network.parameters(), lr = 1e-4)
test_total_loss = []
train_total_loss = []

for epoch in range(1,101):

    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer)
    test_mean_loss, y_true_test, y_pred_test = test_loop(test_data_loader, network, loss_function)
    test_total_loss.append(test_mean_loss)
    train_total_loss.append(train_mean_loss)
    
    if epoch % 1 == 0:
        print(colors.WARNING + f"\nclassification report epoch: " + colors.ENDC + f" {epoch}")

        print(colors.CYAN + f"    train: " + colors.ENDC)
        print(classification_report(
            y_true_train,
            y_pred_train,
            labels = [0, 1, 2, 3],
            target_names=behaviors.keys()
        ))
        plot_confusion_matrix(y_true_train, y_pred_train, behaviors, f"./output/train_confusion_matrix_at_{epoch}.png")

        print(colors.CYAN + f"\n    test: " + colors.ENDC)
        print(classification_report(
            y_true_test,
            y_pred_test,
            labels = [0, 1, 2, 3],
            target_names=behaviors.keys()
        ))
        plot_confusion_matrix(y_true_test, y_pred_test, behaviors, f"./output/test_confusion_matrix_at_{epoch}.png")

        pd.DataFrame(y_pred_test).to_csv(f"./output/y_pred_{epoch}.csv")
        pd.DataFrame(y_true_test).to_csv(f"./output/y_true_{epoch}.csv")

        kernel_heatmap_3d(network.conv3d_1, f"./output/kernel_heatmap_3d_at_{epoch}.png", 0, 0)

loss_over_epochs_lineplot(train_total_loss, f"./output/train_loss_vs_{len(train_total_loss)}_epochs.png")
loss_over_epochs_lineplot(test_total_loss, f"./output/test_loss_vs_{len(test_total_loss)}_epochs.png")

