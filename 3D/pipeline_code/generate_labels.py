import pandas as pd
from natsort import natsorted
import os

def labels(labels_path : str,
           ):

    dictionary = {}

    for name in natsorted(os.listdir(labels_path)):
        full_path = labels_path + "/" + name
        # Skip hidden files and check if it's a directory or CSV file
        if name.startswith('.'):
            continue
        if os.path.isdir(full_path):
            # If it's a directory, look for CSV files inside
            for csv_file in natsorted(os.listdir(full_path)):
                if csv_file.endswith('.csv'):
                    df = pd.read_csv(full_path + "/" + csv_file, index_col=[0])
                    # Use the folder name as video_id to match feature naming
                    dictionary[name] = df.idxmax(axis = 1)
        elif name.endswith('.csv'):
            # If it's a CSV file directly in the path
            df = pd.read_csv(full_path, index_col=[0])
            dictionary[name.replace(".csv","")] = df.idxmax(axis = 1)

    y = pd.concat(dictionary.values(), keys=dictionary.keys(), names=["video_id", "frame"])

    return y
