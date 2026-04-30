import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler


def split_dataframe(X : pd.DataFrame, y : pd.DataFrame, path : str):
    for name in X.index.levels[0]:

        single_vid_X = X.loc[name].iloc[0:18000]
        single_vid_y = y.loc[name].iloc[0:18000]

        single_vid_y[single_vid_y == "background"] = 0
        single_vid_y[single_vid_y == "supportedrear"] = 1
        single_vid_y[single_vid_y == "unsupportedrear"] = 2
        single_vid_y[single_vid_y == "grooming"] = 3

        single_vid_X.to_csv(path + f"/X/{name}.csv")
        single_vid_y.to_csv(path + f"/y/{name}.csv")

class OFTDataset(Dataset):

    def __init__(self, path : str, video_names : list, scaler : StandardScaler,  transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform
        self.path = path
        self.video_names = video_names
        self.scaler = scaler

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        X = pd.read_csv(self.path + f"/X/{video_name}.csv")
        X = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X.to_numpy(dtype = "float32"))
        del X
        y = pd.read_csv(self.path + f"/y/{video_name}.csv")
        y_tensor = torch.from_numpy(y.to_numpy(dtype = "int8"))
        del y
        if self.transform:
            X_tensor = self.transform(X_tensor)
        if self.target_transform:
            y_tensor = self.target_transform(y_tensor)
        return X_tensor, y_tensor

