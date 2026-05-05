from pipeline_code.generate_features_NEW import triangulate, features
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.generate_labels import labels
from sklearn.feature_selection import f_classif, SelectKBest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

X_PATH = f"./pipeline_saved_processes/dataframes/X_3D.csv"
Y_PATH = f"./pipeline_saved_processes/dataframes/y_3D.csv"
COLLECTION_PATH="./pipeline_inputs/collection"
LABELS_PATH = "./pipeline_inputs/labels"


if not (os.path.isfile(X_PATH) and os.path.isfile(Y_PATH)):

    fc = triangulate(collection_path=COLLECTION_PATH)
    X = features(fc, embedding_length = [0])
    y = labels(labels_path = LABELS_PATH)

    X, y = drop_non_analyzed_videos(X=X, y=y)
    X, y = drop_last_frame(X=X, y=y)
    X, y = drop_nas(X=X, y=y)
    X = reduce_bits(X)

    print("saving...")
    X.to_csv(X_PATH)
    y.to_csv(Y_PATH)
    print("!files saved!")

else:

    X = pd.read_csv(X_PATH, index_col=["video_id", "frame"])
    y = pd.read_csv(Y_PATH, index_col=["video_id", "frame"])

selector = SelectKBest(score_func=f_classif, k = "all")
selector.fit(X, y)
scores = selector.scores_
norm_scores = scores / np.max(scores)
feature_importance_dataframe = pd.DataFrame({"feature":pd.Series(X.columns),"score":pd.Series(norm_scores)})
print(feature_importance_dataframe)

plt.figure(figsize = (10,6))
sns.barplot