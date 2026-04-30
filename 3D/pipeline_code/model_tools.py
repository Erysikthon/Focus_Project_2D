import pandas as pd
import random as rd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator

def video_train_test_split(X : pd.DataFrame,y : pd.DataFrame ,test_videos : int ,random_state=None):

    """
    Just like train test split from sklearn, but cooler. \n
    - test_videos : how many videos in the test set?

    returns X_train, X_test, y_train, y_test
    """

    rd.seed(random_state)
    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    if not (X_index.equals(y_index)):
        raise ValueError("X index name doesn't match y index name")
    index = X_index
    test_index = rd.sample(list(index),test_videos)
    train_index = list(index.drop(test_index))
    X_train, X_test, y_train, y_test = X.loc[train_index],X.loc[test_index],y.loc[train_index],y.loc[test_index]
    return X_train, X_test, y_train, y_test

def undersample(X_train : pd.DataFrame, y_train : pd.DataFrame,random_state = None):
    """
    resamples with RandomUnderSampler\n
    - X_train : pd.DataFrame
    - y_train : pd.DataFrame
    - random_state = None | 42 | any
    """
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X_train,y_train)

def smooth_prediction(prediction, min_frames):
    
    #Remove behavior prediction outliers shorter than min_frames.

       # predictions: Array of predicted labels
      #  min_frames: Minimum number of consecutive frames for a behavior to be valid

    #Returns:
      #  Smoothed predictions array
    
    if len(prediction) < min_frames:
        return prediction

    smoothed = prediction.copy()
    i = 0

    while i < len(smoothed):
        current_label = smoothed[i]
        # Find the length of the current behavior segment
        segment_start = i
        while i < len(smoothed) and smoothed[i] == current_label:
            i += 1
        segment_length = i - segment_start

        # If segment is shorter than min_frames, replace with neighboring behavior
        if segment_length < min_frames:
            # Determine replacement label from neighbors
            prev_label = smoothed[segment_start - 1] if segment_start > 0 else None
            next_label = smoothed[i] if i < len(smoothed) else None

            # Choose the label that appears in neighbors (prefer previous)
            if prev_label is not None and next_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label
            else:
                # Keep original if no neighbors available
                continue

            # Replace the short segment
            smoothed[segment_start:i] = replacement

    return smoothed

def predict_multiIndex(model : BaseEstimator, index : list, X : pd.DataFrame, smooth_prediction_frames : int = None):
    predictions_dictionary = {}
    for video_id in index:
        print(f"running prediction for video {video_id}")
        prediction = model.predict(X.loc[video_id])
        if smooth_prediction_frames != None:
            prediction = smooth_prediction(prediction = prediction, min_frames = smooth_prediction_frames)
        predictions_dictionary[video_id] = pd.Series(prediction)
    return pd.DataFrame(pd.concat(predictions_dictionary.values(), keys = predictions_dictionary.keys(), names = ['video_id', 'frame']))

