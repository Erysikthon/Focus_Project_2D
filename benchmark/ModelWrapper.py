import pandas as pd
from utilities import terminal_colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from smoothing_functions import apply_min_duration_filter, apply_gap_fill
from typing import Literal, Sequence, Dict
from CSVReader import csv_read

class ModelWrapper():
    def __init__(self, name : str, test_set : Sequence[str], predictions_folder : str , true_folder : str, output_folder : str, column_names : Dict[int, str], smoothing : Literal["gap", "py3r", "no"] = "gap", smoothing_window : int = 5):

        self.column_names = column_names
        self.name = name
        self.label_wrappers = []
        self.total_prediction = pd.DataFrame()
        self.total_true = pd.DataFrame()
        self.output_folder = output_folder

        for partial_path in test_set:
            total_prediction_file_path = predictions_folder + "/" + partial_path + ".csv"
            total_true_file_path = true_folder + "/" + partial_path + ".csv"
            label_wrapper = LabelWrapper(total_prediction_file_path, total_true_file_path, column_names = column_names, smoothing = smoothing, smoothing_window = smoothing_window)
            self.label_wrappers.append(label_wrapper)
            self.total_prediction = pd.concat((self.total_prediction, label_wrapper.pred), ignore_index = True)
            self.total_true = pd.concat((self.total_true, label_wrapper.true), ignore_index = True)
        
        print(colors.WARNING + f"{name} initialized:\n" +
              colors.CYAN +"   TEST SET SIZE = " + colors.ENDC + f"{len(test_set)}\n"+
              colors.CYAN +"   PREDICTIONS PATH = " + colors.ENDC + f"{predictions_folder}\n"+
              colors.CYAN +"   TRUE PATH = " + colors.ENDC + f"{true_folder}\n" + 
              colors.CYAN +"   OUTPUT PATH = " + colors.ENDC + f"{self.output_folder}\n" +
              colors.CYAN +"   SMOOTHING TYPE = " + colors.ENDC + f"{smoothing}\n" +
              colors.CYAN +"   SMOOTHING WINDOW = " + colors.ENDC + f"{smoothing_window}\n"
              )
        
    def plot_confusion_matrix(self, normalize : bool = True):

        labelnames = ("Background", "Supported Rearing", "Unsupported Rearing", "Grooming", "Digging")
        path = self.output_folder + f"/confusion_matrix_{self.name}.png"
        cm = confusion_matrix(y_true = self.total_true, y_pred = self.total_prediction, labels = list(self.column_names.keys()))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 2)
        
        figure = plt.figure(figsize=(9, 8))
        sns.heatmap(cm, annot=True, cmap='Blues',
                    xticklabels = labelnames,
                    yticklabels = labelnames, 
                    vmin = 0,
                    vmax = 1)
        plt.title(f"Confusion Matrix {self.name}", pad = 10, fontsize = 20, weight = "bold")
        plt.ylabel("True Label", fontsize = 15, labelpad = 10)
        plt.xlabel("Predicted Label", fontsize = 15, labelpad = 10)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(colors.WARNING + f"{self.name} confusion matrix plotted:\n" +
              colors.CYAN +"   NORMALIZE = " + colors.ENDC + f"{normalize}\n"
              )

class LabelWrapper():
    def __init__(self, total_prediction_file_path : str, total_true_file_path : tuple[str], column_names : Dict[str, int], smoothing : Literal["gap", "py3r", "no"], smoothing_window : int = 5):
        self.true, self.pred = csv_read(true_path = total_true_file_path,  pred_path = total_prediction_file_path, cut_and_pad = True, column_names = column_names)

        if smoothing == "gap":
            self.pred = apply_min_duration_filter(self.pred, min_duration = smoothing_window)
            self.pred = apply_gap_fill(self.pred, max_gap = smoothing_window)
            self.pred = pd.Series(self.pred)
        
        if smoothing == "py3r":
            raise KeyError("py3r smoothing COMING SOON")
        
        