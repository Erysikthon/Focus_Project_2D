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
from graphs import f1_over_epochs
from TCNN import TCNN, train_loop, test_loop
from create_video import annotate_video_with_predictions

r = 13
epoch = 0
name = f"TCNN_{epoch}.pt"
video = False
kernels = True
predict = False
debug = False 

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(colors.GREEN + "\nMPS device found\n" + colors.ENDC)
else:
    print (colors.WARNING + "\n[WARNING]: MPS device not found, computing on CPU\n" + colors.ENDC)

video_names = []
for i in range(1,20):
    video_names.append(f"T{i}")

video_names.append("MBT1-M2")
video_names.append("MBT1-M3")
video_names.append("MBT1-M6")
video_names.append("MBT1-M7")
video_names.append("MBT1-M10")
video_names.append("MBT1-M11")
video_names.append("MBT1-M14")
video_names.append("MBT1-M15")
video_names.append("MBT1-M18")

video_names.append("3278_21min_behaviour_2023-01-19T11_08_30")
video_names.append("3279_21min_behaviour_2023-01-19T12_57_29")
video_names.append("BehavioralCamera2023-03-09T10_37_32")
video_names.append("BehavioralCamera2023-03-09T11_04_40")
video_names.append("BehavioralCamera2023-03-09T11_41_07")
video_names.append("BehavioralCamera2023-03-09T12_34_50")

video_names_test = ["T2", "T4", "T13", "MBT1-M2", "MBT1-M7", "MBT1-M10"]
video_names_train = [v for v in video_names if v not in video_names_test]

video_names_test = ["T4"]           ####################     DEBUG        ############################
#video_names_train = ["T4"]           ####################     DEBUG        ############################

features_folder = "./data/rotated_videos"
labels_folder = "./data/labels"
behaviors = {"background" : 0, "Supportedrearing" : 1, "Unsupportedrearing" : 2, "Grooming" : 3, "Digging" : 4}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, s = 1, r = r, n = 40, 
                              undersampling_dict = {"background" : 0.1, "Supportedrearing" : 0.5, "Unsupportedrearing" : 1, "Grooming" : 0.8, "Digging" : 0.3}, 
                              random_state = None, identity = "TRAIN randomized dataset", debug = debug)
test_set = SingleVideoDatasetCollection(features_folder, labels_folder, video_names_test, behaviors,s = 1, r = r, identity = "TEST single dataset collection")

train_data_loader = DataLoader(train_set, 64)
test_data_loader = DataLoader(test_set, 64)

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
    if predict:
        stats = pd.read_csv(f"./output/stats.csv", index_col = 0)
else:
    cols = ["loss_train"]
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_train")
    cols.append("loss_test")
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_test")
    stats = pd.DataFrame(columns = cols)
    stats.to_csv(f"./output/stats.csv")

class_weights = torch.tensor(np.array([1, 1, 1, 1, 0.9], dtype = np.float32)).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(network.parameters(), lr = 0.05)
#torch.nn.init.uniform_(network.fc_4.weight, 0.1, 0.3) 

if kernels and epoch == 0:
    for i in range(0, 5):
        kernel_heatmap_3d(network.conv3d_1, f"./output/conv3d_1_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.conv3d_2, f"./output/conv3d_2_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.conv3d_3, f"./output/conv3d_3_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)

for epoch in range(epoch+1,201):

    if epoch % 3 == 0:
        train_data_loader.dataset.undersample()

    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer, mps_device)

    if predict:
        test_mean_loss, y_true_test, y_pred_test = test_loop(test_data_loader, network, loss_function, mps_device)
    
    if epoch % 1 == 0:
        print(colors.WARNING + f"\nclassification report epoch: " + colors.ENDC + f" {epoch}")

        print(colors.CYAN + f"    train: " + colors.ENDC)
        print(classification_report(y_true_train, y_pred_train, labels = list(behaviors.values()), target_names=behaviors.keys()))
        plot_confusion_matrix(y_true_train, y_pred_train, behaviors, f"./output/train_confusion_matrix_at_{epoch}.png")

        if predict:
            print(colors.CYAN + f"\n    test: " + colors.ENDC)
            print(classification_report(y_true_test, y_pred_test, labels = list(behaviors.values()), target_names=behaviors.keys()))
            plot_confusion_matrix(y_true_test, y_pred_test, behaviors, f"./output/test_confusion_matrix_at_{epoch}.png")

            pd.DataFrame(y_pred_test).to_csv(f"./output/y_pred_{epoch}.csv")
            pd.DataFrame(y_true_test).to_csv(f"./output/y_true_{epoch}.csv")

            row_to_add = [train_mean_loss]
            row_to_add.extend(f1_score(y_true_train, y_pred_train, average = None,  labels = list(behaviors.values())))
            row_to_add.append(test_mean_loss)
            row_to_add.extend(f1_score(y_true_test, y_pred_test, average = None,  labels = list(behaviors.values())))
            stats.loc[epoch] = row_to_add
            stats.to_csv(f"./output/stats.csv")

            loss_over_epochs_lineplot(stats.loc[:, "loss_train"], stats.loc[:, "loss_test"], f"./output/loss_at_{epoch}.png")

            f1_cols = []
            for behavior in behaviors.keys():
                f1_cols.append(f"f1_{behavior}_train")
                f1_cols.append(f"f1_{behavior}_test")
            f1_over_epochs(stats.loc[:, f1_cols], behaviors, f"./output/f1_score_at_{epoch}.png")

        if kernels:
            for i in range(0, 5):
                kernel_heatmap_3d(network.conv3d_1, f"./output/conv3d_1_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.conv3d_2, f"./output/conv3d_2_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.conv3d_3, f"./output/conv3d_3_kernel_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        if video and predict:
            offset = 0
            for dataset in test_data_loader.dataset.collection:
                dataset : SingleVideoDataset
                annotate_video_with_predictions(features_folder + "/" + dataset.file_name + ".mp4", pd.DataFrame(y_pred_test[offset: offset + dataset.get_range() - 1]), 
                                                f"./output/predicted_video_{dataset.file_name}_epoch{epoch}.mp4", (dataset.r - 1)/2, 
                                                pd.DataFrame(y_true_test[offset: offset + dataset.get_range() - 1]))
                offset += dataset.get_range()

        torch.save(network.state_dict(), f"./output/TCNN_{epoch}.pt")
