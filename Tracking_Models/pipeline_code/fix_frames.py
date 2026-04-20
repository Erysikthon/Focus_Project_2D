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

    # Build list of indices to keep (much faster than dropping one by one)
    rows_to_keep_X = []
    rows_to_keep_y = []

    for video_name in index:
        X_video = X.loc[video_name]
        y_video = y.loc[video_name]

        X_len = X_video.shape[0]
        y_len = y_video.shape[0]

        if y_len == X_len:
            # Keep all frames for this video
            if isinstance(X_video.index, pd.MultiIndex):
                rows_to_keep_X.extend([(video_name, frame) for frame in X_video.index])
                rows_to_keep_y.extend([(video_name, frame) for frame in y_video.index])
            else:
                rows_to_keep_X.extend([(video_name, frame) for frame in X_video.index])
                rows_to_keep_y.extend([(video_name, frame) for frame in y_video.index])

        elif y_len > X_len:
            difference = y_len - X_len
            # Keep only first X_len frames from y
            frames_to_keep = y_video.index[:X_len]
            rows_to_keep_y.extend([(video_name, frame) for frame in frames_to_keep])
            rows_to_keep_X.extend([(video_name, frame) for frame in X_video.index])
            print(f"video '{video_name}' has {difference} too many frames in y: dropped {difference}")

        elif y_len < X_len:
            difference = X_len - y_len
            # Keep only first y_len frames from X
            frames_to_keep = X_video.index[:y_len]
            rows_to_keep_X.extend([(video_name, frame) for frame in frames_to_keep])
            rows_to_keep_y.extend([(video_name, frame) for frame in y_video.index])
            print(f"video '{video_name}' has {difference} too many frames in X: dropped {difference}")

    # Filter dataframes by keeping only selected rows
    X = X.loc[rows_to_keep_X]
    y = y.loc[rows_to_keep_y]

    return X, y

def drop_nas(X : pd.DataFrame,y : pd.DataFrame):
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    return X, y
