import pandas as pd
from utilities import terminal_colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class ModelWrapper():
    def __init__(self, name : str, test_set : tuple[str], predictions_path : str , true_path : str, output_path : str):

        self.label_wrappers = []
        self.total_prediction = pd.DataFrame()
        self.total_true = pd.DataFrame()
        self.output_path = output_path
        self.behaviors = {"background" : 0, "Supportedrearing" : 1, "Unsupportedrearing" : 2 , "Grooming" : 3, "Digging" : 4}

        for partial_path in test_set:
            total_prediction_file_path = predictions_path + "/" + partial_path + ".csv"
            total_true_file_path = true_path + "/" + partial_path + ".csv"
            label_wrapper = LabelWrapper(total_prediction_file_path, total_true_file_path, behaviors = self.behaviors)
            self.label_wrappers.append(label_wrapper)
            self.total_prediction = pd.concat((self.total_prediction, label_wrapper.pred), ignore_index = True)
            self.total_true = pd.concat((self.total_true, label_wrapper.true), ignore_index = True)

        self.total_prediction_decoded = one_hot_decoder(self.total_prediction, self.behaviors)
        self.total_true_decoded = one_hot_decoder(self.total_true, self.behaviors)
        
        print(colors.GREEN + f"{name} initialized:\n" +
              colors.CYAN +"   TEST SET SIZE = " + colors.ENDC + f"{len(test_set)}\n"+
              colors.CYAN +"   PREDICTIONS PATH = " + colors.ENDC + f"{predictions_path}\n"+
              colors.CYAN +"   TRUE PATH = " + colors.ENDC + f"{true_path}\n" + 
              colors.CYAN +"   OUTPUT PATH = " + colors.ENDC + f"{self.output_path}\n")
        
    def plot_confusion_matrix(self, normalize : bool = True):

        cm = confusion_matrix(self.total_prediction_decoded, self.total_true_decoded)
        print(cm)

class LabelWrapper():
    def __init__(self, total_prediction_file_path : str, total_true_file_path : tuple[str], behaviors : dict[str, int]):
        self.behaviors = behaviors
        self.pred = pd.read_csv(total_prediction_file_path, index_col = 0)
        self.true = pd.read_csv(total_true_file_path, index_col = 0)
        self.pred_decoded = one_hot_decoder(self.pred, self.behaviors)
        self.true_decoded = one_hot_decoder(self.true, self.behaviors)

def one_hot_decoder(dataframe : pd.DataFrame, behaviors : dict[str, int]):
    
    dataframe = dataframe.copy()
    for behavior in behaviors:
        value = behaviors[behavior]
        dataframe[behavior] = dataframe[behavior]*value
    dataframe = dataframe.sum(axis = 1)
    return dataframe


