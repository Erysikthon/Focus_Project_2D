from torch.nn.modules import Conv3d, ReLU, BatchNorm3d, Module, MaxPool3d, AvgPool3d, Linear, GELU, Flatten, Dropout
from torch.nn import Sequential, ModuleList, CrossEntropyLoss
from torch import Tensor
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utilities import terminal_colors as colors
from tqdm import tqdm
from typing import Sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from VideoDataSet import SingleVideoDataset
from sklearn.metrics import classification_report
import pandas as pd

class ResBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.H = Sequential(
        Conv3d(channels, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        BatchNorm3d(channels),
        ReLU(),
        Conv3d(channels, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        BatchNorm3d(channels)
        )
        self.relu = ReLU()

    def forward(self, x):
        
        return self.relu(self.H(x) + x)

class TCNN(Module):
    def __init__(self):
        super().__init__()
        self.initial_convolution = Sequential(
            Conv3d(1, 32, (5, 5, 5), (2, 2, 2), (2, 2, 2)), 
            BatchNorm3d(32),
            Conv3d(32, 32, (5, 5, 5), (2, 2, 2), (2, 2, 2)), 
            BatchNorm3d(32), 
            ReLU()
        )
        self.block1 = Sequential(*[ResBlock(32) for whatever in range(0, 15)])
        self.transition_12 = Sequential(
            Conv3d(32, 48, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(48),
            Conv3d(48, 48, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(48), 
            MaxPool3d((1, 2, 2), (1, 2, 2)),
            AvgPool3d((2, 1, 1), (2, 1, 1)),
            ReLU()
        )
        self.block2 = Sequential(*[ResBlock(48) for whatever in range(0, 15)])
        self.transition_23 = Sequential(
            Conv3d(48, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(64),
            Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(64), 
            MaxPool3d((1, 2, 2), (1, 2, 2)),
            ReLU()
        )
        self.block3 = Sequential(*[ResBlock(64) for whatever in range(0, 15)])
        self.final_convolution = Sequential(
            Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(64),
            Conv3d(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)), 
            BatchNorm3d(64), 
            ReLU()
        )
        self.time_head = Sequential(
            Linear(2304, 768), 
            ReLU(),
            Dropout(0.3),
            Linear(768, 256), 
            ReLU(), 
            Dropout(0.3)
        )
        self.final_head = Sequential(
            Linear(1024, 512), 
            ReLU(),
            Dropout(0.3),
            Linear(512, 128),
            ReLU(),
            Dropout(0.3),
            Linear(128, 5)
        )

    def forward(self, x : Tensor):
        
        x = self.initial_convolution(x)
        
        x = self.block1(x)
        
        x = self.transition_12(x)
        
        x = self.block2(x)
        
        x = self.transition_23(x)
        
        x = self.block3(x)
        
        x = self.final_convolution(x)
        
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B*T, C*H*W)

        x = self.time_head(x)        # [B*T, 70]
        x = x.view(B, T, 256)

        x = x.reshape(B, -1)       # [B, T*70]
        x = self.final_head(x)
        
        return x

def train_model(dataloaders : Sequence[DataLoader], network : TCNN, loss_fn : CrossEntropyLoss, optimizer : torch.optim.RMSprop, device: torch.device):
    
    network.train()
    total_loss = 0
    with tqdm(desc = colors.CYAN +"    train" + colors.ENDC, total = len(dataloaders), ascii = True) as pbar:
        tot_true = pd.DataFrame()
        tot_pred = pd.DataFrame()
        for dataloader in dataloaders:
            partial_loss = 0
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                pred = network(X)
                pred : Tensor
                tot_pred = pd.concat((tot_pred, pd.DataFrame(pred.detach().argmax(dim = 1).to("cpu").numpy())), axis = 0)
                tot_true = pd.concat((tot_true, pd.DataFrame(y.detach().argmax(dim = 1).to("cpu").numpy())), axis = 0)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                partial_loss += loss.detach().item()
            partial_loss = partial_loss / len(dataloader)
            total_loss += (partial_loss)
            pbar.update(1)
    total_loss = total_loss / len(dataloaders)
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {total_loss}")
    print(classification_report(tot_true, tot_pred))

def test_model(dataloaders : Sequence[DataLoader], network : TCNN, loss_fn : CrossEntropyLoss, device: torch.device, output_folder : str):

    network.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(desc = colors.CYAN +"    test" + colors.ENDC, total = len(dataloaders), ascii = True) as pbar:
            for dataloader in dataloaders:
                output_path = output_folder + "/" + dataloader.dataset.name + ".csv"
                tot_pred = pd.DataFrame()
                partial_loss = 0
                for batch, (X, y) in enumerate(dataloader):
                    X, y = X.to(device), y.to(device)

                    pred = network(X)
                    one_hot = torch.nn.functional.one_hot(pred.argmax(dim=1), num_classes = pred.size(1))
                    loss = loss_fn(pred, y)
                    partial_loss += loss.detach().item()
                    tot_pred = pd.concat((tot_pred, pd.DataFrame(one_hot.to("cpu").numpy())), axis = 0)
                partial_loss = partial_loss / len(dataloader)
                total_loss += (partial_loss)
                tot_pred.columns = ["background", "Supportedrearing", "Unsupportedrearing", "Grooming", "Digging"]
                tot_pred.to_csv(output_path)
                pbar.update(1)
    total_loss = total_loss / len(dataloaders)
    print(colors.WARNING + f"        loss value:" + colors.ENDC + f" {total_loss}")
