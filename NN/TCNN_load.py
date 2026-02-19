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
from sklearn.metrics import classification_report, f1_score
from graphs import kernel_heatmap_3d
from graphs import loss_over_epochs_lineplot
from graphs import plot_confusion_matrix
from TCNN import TCNN, train_loop, test_loop
from create_video import annotate_video_with_predictions

r = 15
epoch = 0
name = f"TCNN_{epoch}.pt"
video = True

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(colors.GREEN + "\nMPS device found\n" + colors.ENDC)
else:
    print (colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + colors.ENDC)

video_names = []
for i in range(1,22):
    if not i==5:
        video_names.append(f"OFT_left_{i}")

features_folder = "./data/rotated_videos"
labels_folder = "./data/labels"
video_names_train, video_names_test = train_test_split(video_names, test_size = 2, shuffle = True, random_state = 42)
behaviors = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, 96, r, 8, random_state = None, identity = "TRAIN randomized dataset")
test_set = SingleVideoDatasetCollection(features_folder, labels_folder, video_names_test, behaviors, 96, r, identity = "TEST single dataset collection")

train_data_loader = DataLoader(train_set, 2)
test_data_loader = DataLoader(test_set, 2)

"""
train_set.__getitem__(0, debug = True)
test_set_collection[0].__getitem__(0, debug = True)
test_set_collection[0].__getitem__(1, debug = True)
"""

network = TCNN().to(mps_device)
print(colors.GREEN + "Network initalized: " + colors.ENDC + f"{network}\n")
if not epoch == 0:
    network.load_state_dict(torch.load(f"./output/{name}"))
    print(colors.CYAN + f"Weights loaded successfully from: " + colors.ENDC + f"{name}")
    stats = pd.read_csv(f"./output/stats_epoch_{epoch}.csv", index_col = 0)
else:
    cols = ["loss_train"]
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_train")
    cols.append("loss_test")
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_test")
    stats = pd.DataFrame(columns = cols)

class_weights = torch.tensor([1.0, 8.9, 40.7, 30.5]).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(network.parameters(), lr = 1e-4)



for epoch in range(epoch+1,101):

    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer, mps_device)
    test_mean_loss, y_true_test, y_pred_test = test_loop(test_data_loader, network, loss_function, mps_device)
    
    if epoch % 1 == 0:
        print(colors.WARNING + f"\nclassification report epoch: " + colors.ENDC + f" {epoch}")

        print(colors.CYAN + f"    train: " + colors.ENDC)
        print(classification_report(y_true_train, y_pred_train, labels = list(behaviors.values()), target_names=behaviors.keys()))
        plot_confusion_matrix(y_true_train, y_pred_train, behaviors, f"./output/train_confusion_matrix_at_{epoch}.png")

        print(colors.CYAN + f"\n    test: " + colors.ENDC)
        print(classification_report(y_true_test, y_pred_test, labels = list(behaviors.values()), target_names=behaviors.keys()))
        plot_confusion_matrix(y_true_test, y_pred_test, behaviors, f"./output/test_confusion_matrix_at_{epoch}.png")

        pd.DataFrame(y_pred_test).to_csv(f"./output/y_pred_{epoch}.csv")
        pd.DataFrame(y_true_test).to_csv(f"./output/y_true_{epoch}.csv")

        kernel_heatmap_3d(network.conv3d_1, f"./output/kernel_heatmap_3d_at_{epoch}.png", 0, 0)

        row_to_add = [train_mean_loss]
        row_to_add.extend(f1_score(y_true_train, y_pred_train, average = None,  labels = list(behaviors.values())))
        row_to_add.append(test_mean_loss)
        row_to_add.extend(f1_score(y_true_test, y_pred_test, average = None,  labels = list(behaviors.values())))
        stats.loc[epoch] = row_to_add
        stats.to_csv(f"./output/stats_epoch_{epoch}.csv")

        loss_over_epochs_lineplot(stats.loc[:, "loss_train"], stats.loc[:, "loss_test"], f"./output/train_loss_vs_{epoch}_epochs.png")

        if video:
            offset = 0
            for dataset in test_data_loader.dataset.collection:
                dataset : SingleVideoDataset
                print(dataset)
                annotate_video_with_predictions(features_folder + "/" + dataset.file_name + ".mp4", pd.DataFrame(y_pred_test[offset: offset + dataset.get_range() - 1]), 
                                                f"./output/predicted_video_{dataset.file_name}_epoch{epoch}.mp4", (dataset.r - 1)/2, 
                                                pd.DataFrame(y_true_test[offset: offset + dataset.get_range() - 1]))
                offset += dataset.get_range()

        torch.save(network.state_dict(), f"./output/TCNN_{epoch}.pt")
