import pandas as pd
from typing import Literal, Sequence, Dict

def csv_read(true_path: str, pred_path: str, cut_and_pad: bool, column_names: dict):

    true_disordered = pd.read_csv(true_path, index_col=0)
    pred_disordered = pd.read_csv(pred_path, index_col=0)
    true = pd.DataFrame()
    pred = pd.DataFrame()

    for i in range(len(column_names)):
        col = column_names[i]
        true[i] = true_disordered[col]
        pred[i] = pred_disordered[col]

    true = true.idxmax(axis=1)
    pred = pred.idxmax(axis=1)

    if cut_and_pad:
        if len(pred) > len(true):
            pred = pred.iloc[:len(true)]
        elif len(pred) < len(true):
            pred = pred.reindex(range(len(true)), fill_value=0)

    return true, pred
