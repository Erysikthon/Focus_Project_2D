import cv2
from torch.utils.data.dataset import Dataset
import torch
from utilities import terminal_colors as colors
from tqdm import tqdm
from torch import Tensor
import pandas as pd
import numpy as np
from typing import Sequence

class SingleVideoDataset(Dataset):
    def __init__(self, video_folder : str, label_folder : str, name : str, window_size : int):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.name = name
        video_path = video_folder + "/" + name + ".mp4"
        label_path = label_folder + "/" + name + ".csv"
        self.window_size = window_size
        self.padding_size = int((window_size - 1)/2)
        cap = cv2.VideoCapture(video_path)
        self.label = pd.read_csv(label_path, index_col = 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames != self.label.shape[0]:
            raise KeyError("video has different length from label")
        
        self.video_array = np.ndarray((frames, self.height, self.width), dtype = np.float32)
        for i in range(0, frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = np.zeros([self.height, self.width]) + 127.5
            self.video_array[i,:,:] = frame
        cap.release()
        padding = np.zeros((self.padding_size, self.height, self.width), dtype = np.float32) + 127.5
        self.video_array = np.concatenate((padding, self.video_array, padding), axis = 0)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index : int):
        X = self.video_array[index : index + self.window_size, :, :]     # video is of shape T, Y, X
        X_tensor = torch.from_numpy((-(X/255 - 0.5) * 2).astype(np.float32))
        X_tensor = X_tensor.unsqueeze(0)       #video is of shape [1, T, Y, X]. Where 1 are the channels, and they will grow with Convolutions
        y = self.label.iloc[index, :]              #label is instead of shape [5] = number of classes one hot encoded
        y_tensor = torch.from_numpy(y.to_numpy().astype(np.float32))
        return X_tensor, y_tensor