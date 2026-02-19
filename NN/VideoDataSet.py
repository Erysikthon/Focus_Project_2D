import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from utilities import terminal_colors as colors
import numpy as np
import cv2
import random as rd
from torch import Tensor

class RandomizedDataset(Dataset):
    def __init__(self, 
                 features_folder : str, 
                 labels_folder : str,
                 file_names : list[str], 
                 behaviors : dict[str,int],
                 s : int,
                 r : int,
                 n : int,
                 random_state = None,
                 identity : str = "randomized dataset"
                 ):
        """
        Docstring for __init__
        
        :param self: Description
        :param features_folder: Description
        :type features_folder: str
        :param labels_folder: Description
        :type labels_folder: str
        :param file_names: Description
        :type file_names: list[str]
        :param behaviors: Description
        :type behaviors: dict[str, int]
        :param s: Snippet size
        :type s: int
        :param r: Receptive field
        :type r: int
        :param n: How many samples per video
        :type n: int
        :param random_state: your random state
        :type random_state: Any
        """

        self.features_folder = features_folder
        self.labels_folder = labels_folder
        self.file_names = file_names
        self.behaviors = behaviors
        self.s = s
        self.r = r
        self.n = n
        rd.seed(random_state)

        print(colors.GREEN + f"{identity} initialized:\n" +
              colors.CYAN +"   videos = " + colors.ENDC + f"{self.file_names}\n"+
              colors.CYAN +"   behaviors = " + colors.ENDC + f"{self.behaviors}\n"+
              colors.CYAN +"   X shape = " + colors.ENDC + f"{self.s + self.r -1}\n"+
              colors.CYAN +"   y shape = " + colors.ENDC + f"{self.s}\n"+
              colors.CYAN +"   N = " + colors.ENDC + f"{self.n}\n"+
              colors.CYAN +"   random state = " + colors.ENDC + f"{random_state}\n")

    def __len__(self):
        return len(self.file_names)*self.n

    def __getitem__(self, index, debug = False):
        file_name = self.file_names[index//self.n]
        video_path = self.features_folder + "/" + file_name + ".mp4"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise KeyError(f"{video_path} is not a valid video path")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        first_n = rd.randint(0, total_frames - (self.s + self.r - 1))

        X = np.ndarray([height, width, self.s + self.r - 1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_n)
        for i in range(0, self.s + self.r - 1):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            X[:,:,i] = frame

            if debug:
                cv2.imshow("Debug Video Frame", frame)
                key = cv2.waitKey(30)
                if key == 27:
                    break

        X_tensor = torch.from_numpy((X/255).astype(np.float32))
        X_tensor = X_tensor.permute(2, 0, 1).unsqueeze(0)

        y_raw = pd.read_csv(self.labels_folder + "/" + file_name + ".csv").iloc[int(first_n + (self.r - 1)/2 ): int(first_n + self.s + (self.r - 1)/2 ), :].reset_index(drop = True)
        y = pd.Series(np.zeros(self.s, dtype = int)-1)
        for behavior in self.behaviors:
            y[y_raw[behavior] == 1] = self.behaviors[behavior]
        if (y == -1).any():
            raise KeyError(f"{file_name} presents a behavior not specified in the behavior list: {self.behaviors}")
        y_tensor = torch.from_numpy(y.to_numpy())

        if debug: 
            print(X_tensor, y_tensor)

        cap.release()
        cv2.destroyAllWindows()
        return X_tensor, y_tensor
        
class SingleVideoDataset(Dataset):
    def __init__(self, 
                 features_folder : str, 
                 labels_folder : str,
                 file_name : str, 
                 behaviors : dict[str,int],
                 s : int,
                 r : int,
                 identity : str = "single video dataset"
                 ):
        """
        Docstring for __init__
        
        :param self: Description
        :param features_folder: Description
        :type features_folder: str
        :param labels_folder: Description
        :type labels_folder: str
        :param file_names: Description
        :type file_names: list[str]
        :param behaviors: Description
        :type behaviors: dict[str, int]
        :param s: Snippet size
        :type s: int
        :param r: Receptive field
        :type r: int
        :param n: How many samples per video
        :type n: int
        :param random_state: your random state
        :type random_state: Any
        """

        self.features_folder = features_folder
        self.labels_folder = labels_folder
        self.file_name = file_name
        self.behaviors = behaviors
        self.s = s
        self.r = r
        self.y_true_total = pd.read_csv(self.labels_folder + "/" + self.file_name + ".csv")

        print(colors.GREEN + f"{identity} initialized:\n" +
              colors.CYAN +"   video = " + colors.ENDC + f"{self.file_name}\n"+
              colors.CYAN +"   behaviors = " + colors.ENDC + f"{self.behaviors}\n"+
              colors.CYAN +"   X shape = " + colors.ENDC + f"{self.s + self.r -1}\n"+
              colors.CYAN +"   y shape = " + colors.ENDC + f"{self.s}\n")

        video_path = self.features_folder + "/" + file_name + ".mp4"

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise KeyError(f"{video_path} is not a valid video path")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __len__(self):
        return (self.total_frames - (self.r - 1)) // self.s

    def __getitem__(self, index, debug = False):

        X = np.ndarray([self.height, self.width, self.s + self.r - 1])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index * self.s)
        for i in range(0, self.s + self.r - 1):
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            X[:,:,i] = frame

            if debug:
                cv2.imshow("Debug Video Frame", frame)
                key = cv2.waitKey(30)
                if key == 27:
                    break

        X_tensor = torch.from_numpy((X/255).astype(np.float32))
        X_tensor = X_tensor.permute(2, 0, 1).unsqueeze(0)

        y_raw = pd.read_csv(self.labels_folder + "/" + self.file_name + ".csv").iloc[int(index * self.s + (self.r - 1)/2) : int((index + 1) * (self.s) + (self.r - 1)/2)].reset_index(drop = True)
        y = pd.Series(np.zeros(self.s, dtype = int) - 1)

        for behavior in self.behaviors:
            y[y_raw[behavior] == 1] = self.behaviors[behavior]
        if (y == -1).any():
            raise KeyError(f"{self.file_name} presents a behavior not specified in the behavior list: {self.behaviors}")
        y_tensor = torch.from_numpy(y.to_numpy())

        if debug: 
            print(X_tensor, y_tensor)

        cv2.destroyAllWindows()
        return X_tensor, y_tensor

class SingleVideoDatasetCollection(SingleVideoDataset):
    def __init__(self, 
                 features_folder : str, 
                 labels_folder : str,
                 file_names : list[str], 
                 behaviors : dict[str,int],
                 s : int,
                 r : int,
                 identity : str = "single video dataset"
                 ):
        """
        Docstring for __init__
        
        :param self: Description
        :param features_folder: Description
        :type features_folder: str
        :param labels_folder: Description
        :type labels_folder: str
        :param file_names: Description
        :type file_names: list[str]
        :param behaviors: Description
        :type behaviors: dict[str, int]
        :param s: Snippet size
        :type s: int
        :param r: Receptive field
        :type r: int
        :param n: How many samples per video
        :type n: int
        :param random_state: your random state
        :type random_state: Any
        """

        self.features_folder = features_folder
        self.labels_folder = labels_folder
        self.file_names = file_names
        self.behaviors = behaviors
        self.s = s
        self.r = r
        self.collection = []
        for file_name in file_names:
            self.collection.append(SingleVideoDataset(features_folder, labels_folder, file_name, behaviors, s, r))

    def __len__(self):
        length = 0
        for item in self.collection:
            length += len(item)
        return length

    def __getitem__(self, index):
        for item in self.collection:
            if index < len(item):
                return item[index]
            index -= len(item)

        raise IndexError("Index out of range")