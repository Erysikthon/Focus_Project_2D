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

r = 15
epoch = 14
name = f"TCNN_{epoch}.pt"
video = True
kernels = True
predict = True
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

video_names.append("20231123_10min_OFT-BL_3919")
video_names.append("20231123_10min_OFT-BL_3961")
video_names.append("20231123_10min_OFT-BL_3962")
video_names.append("20231123_10min_OFT-BL_3963")
video_names.append("20231123_10min_OFT-BL_3964")
video_names.append("20231123_10min_OFT-BL_4025")
video_names.append("20231123_10min_OFT-BL_4028")
video_names.append("20231123_10min_OFT-BL_4029")
video_names.append("BehavioralCamera2023-03-09T12_08_14")
video_names.append("BehavioralCamera2023-03-09T13_02_04")
video_names.append("BehavioralCamera2023-03-09T14_30_45")

video_names.append("BehavioralCamera2023-02-14T13_05_19_shorter")
video_names.append("BehavioralCamera2023-02-14T15_22_37_shorter")
video_names.append("BehavioralCamera2023-02-15T14_40_46_shorter")
video_names.append("BehavioralCamera2023-02-18T10_33_06_shorter")
video_names.append("BehavioralCamera2023-02-18T12_37_43_shorter")
video_names.append("BehavioralCamera2023-02-19T14_53_53_shorter")
video_names.append("BehavioralCamera2023-02-23T10_23_42_shorter")
video_names.append("BehavioralCamera2023-02-23T15_42_37_shorter")
video_names.append("BehavioralCamera2023-02-24T11_06_53_shorter")



video_names_test = [np.str_('3279_21min_behaviour_2023-01-19T12_57_29'), np.str_('20231123_10min_OFT-BL_4028'), np.str_('BehavioralCamera2023-02-23T10_23_42_shorter'), np.str_('MBT1-M2'), np.str_('T2'), 'MBT1-M7', 'T8', 'T4', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'T1']
video_names_train = [v for v in video_names if v not in video_names_test]

video_names_test = ['T1']           ####################     DEBUG        ############################
#video_names_train = ["BehavioralCamera2023-02-24T11_06_53_shorter"]           ####################     DEBUG        ############################

features_folder = "./data/rotated_videos"
labels_folder = "./data/labels"
behaviors = {"background" : 0, "Supportedrearing" : 1, "Unsupportedrearing" : 2, "Grooming" : 3, "Digging" : 4}

train_set = RandomizedDataset(features_folder, labels_folder,  video_names_train, behaviors, s = 1, r = r, n = 5000, 
                              undersampling_dict = {"background" : 0.03, "Supportedrearing" : 0.4, "Unsupportedrearing" : 1, "Grooming" : 0.8, "Digging" : 0.3}, 
                              random_state = None, identity = "TRAIN randomized dataset", debug = debug)
test_set = SingleVideoDatasetCollection(features_folder, labels_folder, video_names_test, behaviors,s = 1, r = r, identity = "TEST single dataset collection")

train_data_loader = DataLoader(train_set, 72)
test_data_loader = DataLoader(test_set, 72)

"""
train_set.__getitem__(0, debug = True)
test_set_collection[0].__getitem__(0, debug = True)
test_set_collection[0].__getitem__(1, debug = True)
"""

network = TCNN().to(mps_device)

print(colors.GREEN + "Network initalized: " + colors.ENDC + f"{network}\n")
total_params = sum(p.numel() for p in network.parameters())
print(colors.CYAN + f"  Total parameters:" + colors.ENDC + f"{total_params:,}")
if not epoch == 0:
    network.load_state_dict(torch.load(f"./output_TCNN/{name}"))
    print(colors.CYAN + f"Weights loaded successfully from: " + colors.ENDC + f"{name}")
    if predict:
        stats = pd.read_csv(f"./output_TCNN/stats.csv", index_col = 0)
else:
    cols = ["loss_train"]
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_train")
    cols.append("loss_test")
    for behavior in behaviors.keys():
        cols.append(f"f1_{behavior}_test")
    stats = pd.DataFrame(columns = cols)
    stats.to_csv(f"./output_TCNN/stats.csv")

class_weights = torch.tensor(np.array([1, 1, 1, 1, 1], dtype = np.float32)).to(mps_device)
loss_function = nn.CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(network.parameters(), 1e-4, weight_decay = 1e-3) # 0.001, weight_decay = 0.01
#torch.nn.init.uniform_(network.fc_4.weight, 0.1, 0.3) 

if kernels and epoch == 0:
    for i in range(0, 5):
        kernel_heatmap_3d(network.initial_convolution[0], f"./output_TCNN/initial_conv_1_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.res_population_1[4].H[0], f"./output_TCNN/res_population_1_nr_9_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.switch_1_2[0], f"./output_TCNN/switch_1_2_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.res_population_2[4].H[0], f"./output_TCNN/res_population_2_nr_9_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        kernel_heatmap_3d(network.final_convolution[3], f"./output_TCNN/final_conv_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)

for epoch in range(epoch+1,2001):

    if epoch >= 0:
        class_weights = torch.tensor(np.array([1, 1, 1, 1, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 0.03, "Supportedrearing" : 0.4, "Unsupportedrearing" : 1, "Grooming" : 0.8, "Digging" : 0.3}

    if epoch >= 20:
        class_weights = torch.tensor(np.array([1, 1, 1.3, 1, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 0.2, "Supportedrearing" : 0.7, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 0.45}

    if epoch >= 40:
        class_weights = torch.tensor(np.array([0.5, 1, 1.5, 1, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 0.4, "Supportedrearing" : 0.7, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 0.6}

    if epoch >= 60:
        class_weights = torch.tensor(np.array([0.25, 1, 2.5, 1.2, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 0.7, "Supportedrearing" : 1, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 0.8}

    if epoch >= 80:
        class_weights = torch.tensor(np.array([0.5, 1.3, 4.5, 2.8, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 1, "Supportedrearing" : 1, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 1}

    if epoch >= 100:
        class_weights = torch.tensor(np.array([0.5, 1.3, 4.5, 2.8, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 1, "Supportedrearing" : 1, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 1}
        optimizer = torch.optim.AdamW(network.parameters(), 5e-5, weight_decay = 5e-4)

    if epoch >= 120:
        class_weights = torch.tensor(np.array([0.5, 1.3, 4.5, 2.8, 1], dtype = np.float32)).to(mps_device)
        train_data_loader.dataset.undersampling_dict = undersampling_dict = {"background" : 1, "Supportedrearing" : 1, "Unsupportedrearing" : 1, "Grooming" : 1, "Digging" : 1}
        optimizer = torch.optim.AdamW(network.parameters(), 1e-5, weight_decay = 1e-4)

    if epoch % 3 == 0:
        train_data_loader.dataset.undersample()

    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_mean_loss, y_true_train, y_pred_train  = train_loop(train_data_loader, network, loss_function, optimizer, mps_device)
    
    if epoch % 1 == 0:
        print(colors.WARNING + f"\nclassification report epoch: " + colors.ENDC + f" {epoch}")

        print(colors.CYAN + f"    train: " + colors.ENDC)
        print(classification_report(y_true_train, y_pred_train, labels = list(behaviors.values()), target_names=behaviors.keys()))
        plot_confusion_matrix(y_true_train, y_pred_train, behaviors, f"./output_TCNN/train_confusion_matrix_at_{epoch}.png")

        if kernels:
            for i in range(0, 5):
                kernel_heatmap_3d(network.initial_convolution[0], f"./output_TCNN/initial_conv_1_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.res_population_1[4].H[0], f"./output_TCNN/res_population_1_nr_9_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.switch_1_2[0], f"./output_TCNN/switch_1_2_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.res_population_2[4].H[0], f"./output_TCNN/res_population_2_nr_9_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
                kernel_heatmap_3d(network.final_convolution[3], f"./output_TCNN/final_conv_at_{i}_heatmap_3d_at_{epoch}.png", i, 0)
        
    if epoch % 50 == 0 and predict:
        test_mean_loss, y_true_test, y_pred_test = test_loop(test_data_loader, network, loss_function, mps_device)

        print(colors.CYAN + f"\n    test: " + colors.ENDC)
        print(classification_report(y_true_test, y_pred_test, labels = list(behaviors.values()), target_names=behaviors.keys()))
        plot_confusion_matrix(y_true_test, y_pred_test, behaviors, f"./output_TCNN/test_confusion_matrix_at_{epoch}.png")

        pd.DataFrame(y_pred_test).to_csv(f"./output_TCNN/y_pred_{epoch}.csv")
        pd.DataFrame(y_true_test).to_csv(f"./output_TCNN/y_true_{epoch}.csv")

        row_to_add = [train_mean_loss]
        row_to_add.extend(f1_score(y_true_train, y_pred_train, average = None,  labels = list(behaviors.values())))
        row_to_add.append(test_mean_loss)
        row_to_add.extend(f1_score(y_true_test, y_pred_test, average = None,  labels = list(behaviors.values())))
        stats.loc[epoch] = row_to_add
        stats.to_csv(f"./output_TCNN/stats.csv")

        loss_over_epochs_lineplot(stats.loc[:, "loss_train"], stats.loc[:, "loss_test"], f"./output_TCNN/loss_at_{epoch}.png")

        f1_cols = []
        for behavior in behaviors.keys():
            f1_cols.append(f"f1_{behavior}_train")
            f1_cols.append(f"f1_{behavior}_test")
        f1_over_epochs(stats.loc[:, f1_cols], behaviors, f"./output_TCNN/f1_score_at_{epoch}.png")

        if video:
            offset = 0
            for dataset in test_data_loader.dataset.collection:
                dataset : SingleVideoDataset
                annotate_video_with_predictions(features_folder + "/" + dataset.file_name + ".mp4", pd.DataFrame(y_pred_test[offset: offset + dataset.get_range() - 1]), 
                                                f"./output_TCNN/predicted_video_{dataset.file_name}_epoch{epoch}.mp4", (dataset.r - 1)/2, 
                                                pd.DataFrame(y_true_test[offset: offset + dataset.get_range() - 1]))
                offset += dataset.get_range()

    torch.save(network.state_dict(), f"./output_TCNN/TCNN_{epoch}.pt")
