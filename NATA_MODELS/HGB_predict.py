"""
HGB Prediction Script

Loads a trained HistGradientBoosting model, runs inference on the test set,
and saves per-frame predicted labels as one CSV per video.

"""

import numpy as np
import pandas as pd
import os
import joblib


# ============================================================================
# Configuration
# ============================================================================

X_FILTERED_PATH = "./pipeline_saved_processes/dataframes/X_filtered.csv"
Y_PATH          = "./pipeline_saved_processes/dataframes/y.csv"
MODEL_SAVE_PATH = "./pipeline_saved_processes/models/HGB_HGB_v1.pkl"
OUTPUT_FOLDER   = "./pipeline_outputs/HGB/predictions"

test_video_ids = [
    '20231123_10min_OFT-BL_4025', '3279_21min_behaviour_2023-01-19T12_57_29',
    'BehavioralCamera2023-02-23T10_23_42_shorter',
    'BehavioralCamera2023-02-24T11_06_53_shorter',
    'BehavioralCamera2023-03-09T12_08_14', 'MBT1-M7', 'T11', 'T15', 'T4', 'T6'
]


# ============================================================================
# Load videos
# ============================================================================

print("Loading features and labels...")
X = pd.read_csv(X_FILTERED_PATH, index_col=["video_id", "frame"])
y = pd.read_csv(Y_PATH, index_col=["video_id", "frame"])

all_video_ids  = X.index.get_level_values("video_id").unique().tolist()
test_video_ids = [v for v in test_video_ids if v in all_video_ids]

missing = set(test_video_ids) - set(all_video_ids)
if missing:
    print(f"Warning: test videos not found in data: {missing}")

print(f"Test videos: {len(test_video_ids)}")

# Derive behavior names from y (sorted, lowercase)
behavior_names = sorted(y.iloc[:, 0].unique().tolist())
print(f"Behaviors: {behavior_names}")

# ============================================================================
# Load model
# ============================================================================

print(f"\nLoading model from {MODEL_SAVE_PATH}...")
model = joblib.load(MODEL_SAVE_PATH).model
print("Model loaded.")


# ============================================================================
# Predict and save per-video CSVs
# ============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for video_id in test_video_ids:
    X_video = X.loc[video_id]           # DataFrame: index=frame, cols=features
    frame_indices = X_video.index.tolist()

    preds = model.predict(X_video)      # array of class name strings

    # One-hot encode
    one_hot = np.zeros((len(preds), len(behavior_names)), dtype=int)
    for i, cls_name in enumerate(preds):
        if cls_name in behavior_names:
            one_hot[i, behavior_names.index(cls_name)] = 1

    display_names = [b if b == "background" else b.capitalize() for b in behavior_names]
    df = pd.DataFrame(one_hot, columns=display_names, index=frame_indices)
    df.index.name = "frame"

    out_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")
    df.to_csv(out_path)
    print(f"Saved: {out_path}  ({len(df)} frames)")

print(f"\nDone. Predictions saved to: {OUTPUT_FOLDER}")
