import pandas as pd

def drop_non_analyzed_videos(X : pd.DataFrame,y : pd.DataFrame):
    X_videos = X.index.get_level_values("video_id").unique()
    y_videos = y.index.get_level_values("video_id").unique()

    # Keep only videos that exist in both X and y
    common_videos = X_videos.intersection(y_videos)

    X = X.loc[common_videos]
    y = y.loc[common_videos]

    return X, y

def drop_last_frame(X : pd.DataFrame,y : pd.DataFrame):
    X_index = X.index.get_level_values("video_id").unique()
    y_index = y.index.get_level_values("video_id").unique()
    if not (X_index.equals(y_index)):
        raise ValueError("X index name doesn't match y index name")
    index = X_index
    for video_name in index:
        if y.loc[video_name].shape[0] == X.loc[video_name].shape[0]:
            continue
        
        elif y.loc[video_name].shape[0] > X.loc[video_name].shape[0]:
            difference = y.loc[video_name].shape[0] - X.loc[video_name].shape[0]
            for i in range(0,difference):
                y = y.drop((video_name, y.loc[video_name].index[-1]))
            print(f"video '{video_name}' has {difference} too many frames in y: dropped {difference}")

        elif y.loc[video_name].shape[0] < X.loc[video_name].shape[0]:
            difference = X.loc[video_name].shape[0] - y.loc[video_name].shape[0]
            for i in range(0,difference):
                X = X.drop((video_name, X.loc[video_name].index[-1]))
            print(f"video '{video_name}' has {difference} too many frames in X: dropped {difference}")

    return X, y

def drop_nas(X : pd.DataFrame,y : pd.DataFrame):
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    return X, y
