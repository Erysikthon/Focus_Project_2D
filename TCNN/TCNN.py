import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from utilities import terminal_colors as colors
from tqdm import tqdm
from torch import Tensor
from VideoDataSet import SingleVideoDataset
import seaborn as sns
import matplotlib.pyplot as plt
from TCNN_model import TCNN, train_model, test_model
from torch.nn.modules import Conv3d, ReLU, BatchNorm3d, Module, MaxPool3d, AvgPool3d, Linear, GELU, Flatten
from torch.nn import Sequential, ModuleList, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

VIDEO_FOLDER = "./data/rotated_videos"
LABEL_FOLDER = "./data/labels"
OUTPUT_FOLDER = "./output"
WINDOW_SIZE = 31
BATCH_SIZE = 64
TRAIN_VIDEO_NAMES = ['20231123_10min_OFT-BL_3961', '20231123_10min_OFT-BL_3962', '20231123_10min_OFT-BL_3963', '20231123_10min_OFT-BL_3964', '20231123_10min_OFT-BL_4028', '3278_21min_behaviour_2023-01-19T11_08_30', 'BehavioralCamera2023-02-14T13_05_19_shorter', 'BehavioralCamera2023-02-14T15_22_37_shorter', 'BehavioralCamera2023-02-15T14_40_46_shorter', 'BehavioralCamera2023-02-18T10_33_06_shorter', 'BehavioralCamera2023-02-18T12_37_43_shorter', 'BehavioralCamera2023-02-23T15_42_37_shorter', 'BehavioralCamera2023-03-09T10_37_32', 'BehavioralCamera2023-03-09T11_04_40', 'BehavioralCamera2023-03-09T11_41_07', 'BehavioralCamera2023-03-09T12_34_50', 'BehavioralCamera2023-03-09T13_02_04', 'MBT1-M10', 'MBT1-M11', 'MBT1-M15', 'MBT1-M18', 'MBT1-M2', 'MBT1-M6', 'T1', 'T12', 'T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T2', 'T5', 'T8', 'T9']
TEST_VIDEO_NAMES = ['20231123_10min_OFT-BL_4025', '3279_21min_behaviour_2023-01-19T12_57_29', 'BehavioralCamera2023-02-23T10_23_42_shorter', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'BehavioralCamera2023-03-09T12_08_14', 'MBT1-M7', 'T11', 'T15', 'T4', 'T6']


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(colors.GREEN + "\nMPS device found\n" + colors.ENDC)
elif torch.cuda.is_available(): 
    device = torch.device("cuda")
    print(colors.GREEN + "\nCUDA device found\n" + colors.ENDC)
else:
    print (colors.WARNING + "\n[WARNING]: accelerator not found, computing on CPU\n" + colors.ENDC)
    device = torch.device("cpu")

network = TCNN().to(device)
class_weights = torch.tensor(np.array([0.3, 1.3, 4.5, 2.8, 1], dtype = np.float32)).to(device)
loss = CrossEntropyLoss(class_weights)
optimizer = torch.optim.AdamW(params = network.parameters(), lr = 0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)

train_dataloaders = []
for name in TRAIN_VIDEO_NAMES:
    train_dataloaders.append(DataLoader(SingleVideoDataset(video_folder = VIDEO_FOLDER, label_folder = LABEL_FOLDER, name = name, window_size = WINDOW_SIZE), 
                                        batch_size = BATCH_SIZE))

test_dataloaders = []
for name in TEST_VIDEO_NAMES:
    test_dataloaders.append(DataLoader(SingleVideoDataset(video_folder = VIDEO_FOLDER, label_folder = LABEL_FOLDER, name = name, window_size = WINDOW_SIZE),
                                        batch_size = BATCH_SIZE))

for epoch in range(1, 21):
    print(colors.GREEN + f"\nEpoch:" + colors.ENDC + f" {epoch}")
    train_model(dataloaders = train_dataloaders, network = network, loss_fn = loss, optimizer = optimizer, device = device)
    test_model(dataloaders = test_dataloaders, network = network, loss_fn = loss, device = device, output_folder = OUTPUT_FOLDER)
    scheduler.step()
    print(colors.WARNING + f"Learning Rate: " + colors.ENDC + f"{optimizer.param_groups[0]["lr"]}")

torch.save(network.state_dict(), OUTPUT_FOLDER + "/TCNN.pt")