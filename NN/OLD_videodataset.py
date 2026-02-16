import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from utilities import terminal_colors
import numpy as np
from sklearn.impute import SimpleImputer

class VideoDataset(Dataset):

    def __init__(self, path : str, video_names : list, scaler : StandardScaler, frames : int = 18000, 
                 behaviors : dict = {"background" : 0, "supportedrear" : 1, "unsupportedrear" : 2, "grooming" : 3}, 
                 transform = None, target_transform = None, 
                 points : list = ['mouse_top.mouse_top_0.nose.x', 'mouse_top.mouse_top_0.nose.y',
                                  'mouse_top.mouse_top_0.headcentre.x','mouse_top.mouse_top_0.headcentre.y',
                                  'mouse_top.mouse_top_0.neck.x','mouse_top.mouse_top_0.neck.y', 
                                  'mouse_top.mouse_top_0.earl.x', 'mouse_top.mouse_top_0.earl.y',
                                  'mouse_top.mouse_top_0.earr.x','mouse_top.mouse_top_0.earr.y', 
                                  'mouse_top.mouse_top_0.bodycentre.x','mouse_top.mouse_top_0.bodycentre.y',
                                  'mouse_top.mouse_top_0.bcl.x','mouse_top.mouse_top_0.bcl.y',
                                  'mouse_top.mouse_top_0.bcr.x', 'mouse_top.mouse_top_0.bcr.y',
                                  'mouse_top.mouse_top_0.hipl.x','mouse_top.mouse_top_0.hipl.y', 
                                  'mouse_top.mouse_top_0.hipr.x', 'mouse_top.mouse_top_0.hipr.y',
                                  'mouse_top.mouse_top_0.tailbase.x','mouse_top.mouse_top_0.tailbase.y',
                                  'mouse_top.mouse_top_0.tailcentre.x','mouse_top.mouse_top_0.tailcentre.y',
                                  'mouse_top.mouse_top_0.tailtip.x', 'mouse_top.mouse_top_0.tailtip.y',
                                  'oft_3d.oft_3d_0.tl.x', 'oft_3d.oft_3d_0.tl.y',
                                  'oft_3d.oft_3d_0.tr.x','oft_3d.oft_3d_0.tr.y', 
                                  'oft_3d.oft_3d_0.bl.x', 'oft_3d.oft_3d_0.bl.y',
                                  'oft_3d.oft_3d_0.br.x','oft_3d.oft_3d_0.br.y',
                                  'oft_3d.oft_3d_0.top_tl.x', 'oft_3d.oft_3d_0.top_tl.y',
                                  'oft_3d.oft_3d_0.top_tr.x','oft_3d.oft_3d_0.top_tr.y',
                                  'oft_3d.oft_3d_0.top_bl.x', 'oft_3d.oft_3d_0.top_bl.y',
                                  'oft_3d.oft_3d_0.top_br.x','oft_3d.oft_3d_0.top_br.y', ],
                 identity : str = "dataset"):
        
        self.transform = transform
        self.target_transform = target_transform
        self.path = path
        self.video_names = video_names
        self.scaler = scaler
        self.frames = frames
        self.behaviors = behaviors
        self.points = points
        self.identity = identity

        print(terminal_colors.GREEN + f"{identity} initialized:\n" +
              terminal_colors.CYAN +"   videos = " + terminal_colors.ENDC + f"{self.video_names}\n"+
              terminal_colors.CYAN +"   scaler = " + terminal_colors.ENDC + f"{self.scaler}\n" +
              terminal_colors.CYAN +"   behaviors = " + terminal_colors.ENDC + f"{self.behaviors}\n"+
              terminal_colors.CYAN +"   frames = " + terminal_colors.ENDC + f"{self.frames}\n"+
              terminal_colors.CYAN +"   points = " + terminal_colors.ENDC + f"{self.points}\n")


    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        X = pd.read_csv(self.path + f"/features/{video_name}")
        imputer = SimpleImputer(strategy = "mean").set_output(transform = "pandas")
        X = imputer.fit_transform(X)
        X = self.scaler.transform(X)
        X = X.loc[0:self.frames-1, self.points]
        X_tensor = torch.from_numpy(X.to_numpy(dtype = "float32"))
        
        y_raw = pd.read_csv(self.path + f"/labels/{video_name}")[0:self.frames]
        y = pd.Series(np.zeros(self.frames, dtype = int)-1)
        for behavior in self.behaviors:
            y[y_raw[behavior] == 1] = self.behaviors[behavior]

        if (y == -1).any():
            raise KeyError(terminal_colors.FAIL + f"{video_name} presents a behavior not specified in the behavior list: {self.behaviors}"+ terminal_colors.ENDC)

        y_tensor = torch.from_numpy(y.to_numpy())

        if self.transform:
            X_tensor = self.transform(X_tensor)
        if self.target_transform:
            y_tensor = self.target_transform(y_tensor)
        return X_tensor, y_tensor

