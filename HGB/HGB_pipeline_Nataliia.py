import os
import itertools
import numpy as np
import pandas as pd
from natsort import natsorted

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report

from pipeline_code.generate_features import features_2d
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos, drop_last_frame, drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits, collinearity_filter

from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection


OUTPUT_DIR = "./pipeline_outputs/hgb_simple"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# SPLIT
# =========================================================
train_video_ids = [
    '20231123_10min_OFT-BL_3961', '20231123_10min_OFT-BL_3962',
    '20231123_10min_OFT-BL_3963', '20231123_10min_OFT-BL_3964',
    '20231123_10min_OFT-BL_4028', '3278_21min_behaviour_2023-01-19T11_08_30',
    'BehavioralCamera2023-02-14T13_05_19_shorter',
    'BehavioralCamera2023-02-14T15_22_37_shorter',
    'BehavioralCamera2023-02-15T14_40_46_shorter',
    'BehavioralCamera2023-02-18T10_33_06_shorter',
    'BehavioralCamera2023-02-18T12_37_43_shorter',
    'BehavioralCamera2023-02-23T15_42_37_shorter',
    'BehavioralCamera2023-03-09T10_37_32',
    'BehavioralCamera2023-03-09T11_04_40',
    'BehavioralCamera2023-03-09T11_41_07',
    'BehavioralCamera2023-03-09T12_34_50',
    'BehavioralCamera2023-03-09T13_02_04',
    'MBT1-M10', 'MBT1-M11', 'MBT1-M15', 'MBT1-M18', 'MBT1-M2', 'MBT1-M6',
    'T1', 'T12', 'T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T2', 'T5', 'T8', 'T9'
]

val_video_ids = [
    '20231123_10min_OFT-BL_3919', '20231123_10min_OFT-BL_4029',
    'BehavioralCamera2023-02-19T14_53_53_shorter',
    'BehavioralCamera2023-03-09T14_30_45',
    'MBT1-M14', 'MBT1-M3', 'T10', 'T3', 'T7'
]

test_video_ids = [
    '20231123_10min_OFT-BL_4025',
    '3279_21min_behaviour_2023-01-19T12_57_29',
    'BehavioralCamera2023-02-23T10_23_42_shorter',
    'BehavioralCamera2023-02-24T11_06_53_shorter',
    'BehavioralCamera2023-03-09T12_08_14',
    'MBT1-M7', 'T11', 'T15', 'T4', 'T6'
]


# =========================================================
# BUILD X AND y
# =========================================================
collection_path = "./pipeline_inputs/collection"
fps = 30
filter_threshold = 0.9
rescale_points = ("tr", "tl")
rescale_distance_mbt = 0.47
rescale_distance_default = 0.64

tracking_dict = {}
csv_files = natsorted([
    f for f in os.listdir(collection_path)
    if f.endswith(".csv") and not f.startswith(".")
])

for csv_file in csv_files:
    video_handle = os.path.splitext(csv_file)[0]
    csv_path = os.path.join(collection_path, csv_file)
    tracking_dict[video_handle] = Tracking.from_yolo3r(
        filepath=csv_path,
        handle=video_handle,
        fps=fps
    )

tracking_collection = TrackingCollection(tracking_dict)
tracking_collection.each.strip_column_names()

videos_to_remove = []
for video_id, tracking in tracking_collection._obj_dict.items():
    required_columns = ['tr.x', 'tr.y', 'tl.x', 'tl.y', 'br.x', 'br.y', 'bl.x', 'bl.y']
    if not all(col in tracking.data.columns for col in required_columns):
        videos_to_remove.append(video_id)
        print(f"Warning: {video_id} missing OFT corners -> excluded")

for video_id in videos_to_remove:
    del tracking_collection._obj_dict[video_id]

tracking_collection.each.filter_likelihood(filter_threshold)

for video_id, tracking in tracking_collection._obj_dict.items():
    if "MBT" in video_id:
        tracking.rescale_by_known_distance(
            rescale_points[0], rescale_points[1], rescale_distance_mbt, dims=("x", "y")
        )
    else:
        tracking.rescale_by_known_distance(
            rescale_points[0], rescale_points[1], rescale_distance_default, dims=("x", "y")
        )

features_collection = FeaturesCollection.from_tracking_collection(tracking_collection)

X = features_2d(
    features_collection,
    distance={
        ("neck", "earl"): ("x", "y"),
        ("neck", "earr"): ("x", "y"),
        ("neck", "bcl"): ("x", "y"),
        ("neck", "bcr"): ("x", "y"),
        ("bcl", "hipl"): ("x", "y"),
        ("bcr", "hipr"): ("x", "y"),
        ("hipl", "tailbase"): ("x", "y"),
        ("hipr", "tailbase"): ("x", "y"),
        ("headcentre", "neck"): ("x", "y"),
        ("neck", "bodycentre"): ("x", "y"),
        ("bodycentre", "tailbase"): ("x", "y"),
        ("headcentre", "earl"): ("x", "y"),
        ("headcentre", "earr"): ("x", "y"),
        ("bodycentre", "bcl"): ("x", "y"),
        ("bodycentre", "bcr"): ("x", "y"),
        ("bodycentre", "hipl"): ("x", "y"),
        ("bodycentre", "hipr"): ("x", "y")
    },
    azimuth_deviation={
        ("neck", "bcl", "earl"),
        ("neck", "earr", "bcr"),
        ("bodycentre", "hipl", "bcl"),
        ("bodycentre", "bcr", "hipr"),
        ("bodycentre", "tailbase", "hipl"),
        ("bodycentre", "hipr", "tailbase")
    },
    azimuth={
        ("bodycentre", "neck"),
        ("neck", "earl"),
        ("neck", "earr"),
        ("tailbase", "bodycentre"),
        ("tailbase", "hipr"),
        ("tailbase", "hipl"),
        ("neck", "nose")
    },
    speed=(
        "headcentre", "earl", "earr", "neck", "bcl",
        "bcr", "bodycentre", "hipl", "hipr", "tailcentre"
    ),
    distance_change=(
        "headcentre", "earl", "earr", "neck", "bcl",
        "bcr", "bodycentre", "hipl", "hipr", "tailcentre"
    ),
    area_of_boundary={
        ("nose", "earl", "neck", "earr"),
        ("neck", "bcr", "hipr", "tailbase", "hipl", "bcl")
    },
    distance_to_boundary=(
        "headcentre", "earl", "earr", "neck", "bcl",
        "bcr", "bodycentre", "hipl", "hipr", "tailcentre"
    ),
    f_b_fill=True,
    embedding_length=list(range(-15, 16, 1))
)

y = labels(labels_path="./pipeline_inputs/labels")

X, y = drop_non_analyzed_videos(X=X, y=y)
X, y = drop_last_frame(X=X, y=y)
X, y = drop_nas(X=X, y=y)

X = reduce_bits(X)
X = collinearity_filter(X, threshold=0.95)


# =========================================================
# TRAIN / VAL / TEST SPLIT
# =========================================================
video_ids = X.index.get_level_values("video_id").astype(str)

X_train = X.loc[video_ids.isin(train_video_ids)]
X_val = X.loc[video_ids.isin(val_video_ids)]
X_test = X.loc[video_ids.isin(test_video_ids)]

y_train = y.loc[video_ids.isin(train_video_ids)]
y_val = y.loc[video_ids.isin(val_video_ids)]
y_test = y.loc[video_ids.isin(test_video_ids)]

test_video_ids_per_row = np.array(X_test.index.get_level_values("video_id").astype(str))
test_frames_per_row = np.array(X_test.index.get_level_values("frame"))

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = np.array(y_train).ravel()
y_val = np.array(y_val).ravel()
y_test = np.array(y_test).ravel()


# =========================================================
# SIMPLE HGB MODEL SELECTION ON VALIDATION
# =========================================================
param_grid = {
    "max_depth": [6, 7],
    "min_samples_leaf": [40, 50],
    "learning_rate": [0.1],
    "max_iter": [300]
}

best_model = None
best_params = None
best_val_f1 = -1.0

for max_depth, min_samples_leaf, learning_rate, max_iter in itertools.product(
    param_grid["max_depth"],
    param_grid["min_samples_leaf"],
    param_grid["learning_rate"],
    param_grid["max_iter"]
):
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        max_iter=max_iter
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    print(
        f"Params: max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}, "
        f"learning_rate={learning_rate}, "
        f"max_iter={max_iter} "
        f"-> val macro F1={val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model
        best_params = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
            "max_iter": max_iter
        }

print("\nBest params selected on validation:")
print(best_params)
print(f"Best validation macro F1: {best_val_f1:.4f}")


# =========================================================
# TEST EVALUATION
# =========================================================
test_pred = best_model.predict(X_test)
test_macro_f1 = f1_score(y_test, test_pred, average="macro")
test_weighted_f1 = f1_score(y_test, test_pred, average="weighted")

print("\n=== TEST RESULTS ===")
print(f"Test macro F1:    {test_macro_f1:.4f}")
print(f"Test weighted F1: {test_weighted_f1:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, test_pred, digits=4))


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================
df_test_predictions = pd.DataFrame({
    "video_id": test_video_ids_per_row,
    "frame": test_frames_per_row,
    "y_true": y_test,
    "y_pred": test_pred
})

df_test_predictions = df_test_predictions.sort_values(["video_id", "frame"]).reset_index(drop=True)
df_test_predictions.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)

print(f"\nSaved test predictions to: {OUTPUT_DIR}/test_predictions.csv")