import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix

def kernel_heatmap(conv_layer : torch.nn.modules.Conv1d, output_path : str, n_kernels):

    kernels = conv_layer.weight.detach().cpu()

    y_max = n_kernels//6
    x_max = 6

    figure, ax = plt.subplots(y_max, x_max, figsize = (50,30))
    figure.suptitle(f"kernels of conv1d layer: {conv_layer}")

    for i in range(0, n_kernels):
        axis = ax[i//x_max][i%x_max]
        sns.heatmap(kernels[i], ax = axis, cmap = "viridis", vmin = -0.25, vmax = 0.25)
        axis.set_title(f"kernel: {i}")
        

    plt.savefig(output_path)

def loss_over_epochs_lineplot(total_loss,output_path : str):
    epoch = len(total_loss)
    figure = plt.figure(figsize = (10,6))
    sns.lineplot(y = total_loss, x = np.arange(0,epoch))
    plt.title(f"loss vs epochs")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(output_path)
    plt.close(figure)
    
def plot_confusion_matrix(y_true, y_pred, behaviors : dict, output_path : str):
        
        behavior_numbers = list(behaviors.values())
        behavior_names = list(behaviors.keys())
        print("Labels order:", np.unique(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=behavior_numbers)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Print with labels for debugging
        print(cm)

        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cmn,
            annot=True,           # Show numbers in cells
            fmt='.2f',              # Format as percentages
            cmap='Blues',         # Color scheme
            xticklabels=behavior_names,   # Label x-axis with class names in correct order
            yticklabels=behavior_names,   # Label y-axis with class names in correct order
            cbar_kws={'label': 'Proportion'}
        )
        plt.title('Confusion Matrix ', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, labelpad=15)
        plt.xlabel('Predicted Label', fontsize=12, labelpad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
