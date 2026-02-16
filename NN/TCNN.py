import torch
import numpy as np
import pandas as pd
import os
from utilities import terminal_colors as colors
from VideoDataSet import RandomizedDataset, SingleVideoDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn

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
video_names_train, video_names_test = train_test_split(video_names, test_size = 4, shuffle = True, random_state = 42)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, 98, 31, 60, random_state = 42, identity = "TRAIN randomized dataset")
test_set_collection = []
for i in range(0, len(video_names_test)):
    test_set_collection.append(SingleVideoDataset(features_folder, labels_folder, video_names_test[i], behaviors, 98, 31, f"TEST single video {i} dataset"))


train_data_loader = DataLoader(train_set)

"""
train_set.__getitem__(0, debug = True)
test_set_collection[0].__getitem__(0, debug = True)
test_set_collection[0].__getitem__(1, debug = True)
"""

class TCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 16, (5, 5, 5), 1, padding = 0)
        self.relu_1 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm3d(16, 0.001)

        self.conv3d_2 = nn.Conv3d(16, 64, (7, 7, 7), 1, padding = 0)
        self.relu_2 = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm3d(64, 0.001)

        self.conv3d_3 = nn.Conv3d(64, 128, (5, 5, 5), 1, padding = 0)
        self.relu_3 = nn.ReLU()
        self.batchnorm_3 = nn.BatchNorm3d(64, 0.001)


    def forward(self, x):
        x = x.unsqueeze(0)

        x = self.conv3d_1(x)
        x = self.relu_1(x)
        x = self.batchnorm_1(x)

        x = self.conv3d_2(x)
        x = self.relu_2(x)
        x = self.batchnorm_2(x)

        print(x.shape)

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

class_weights = torch.tensor([1.0, 4, 10, 10]).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.RMSprop(network.parameters(), lr = 1e-4)
test_total_loss = []
train_total_loss = []

for epoch in range(1,101):

    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer)
