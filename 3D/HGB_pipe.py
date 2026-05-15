from pipeline_code.generate_features_NEW import triangulate
from pipeline_code.generate_features_NEW import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.filter_and_preprocess import collinearity_filter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_code.Shelf import Shelf
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from pipeline_code.model_tools import predict_multiIndex
from sklearn.model_selection import GridSearchCV, GroupKFold
import joblib
import time
import pandas as pd
from natsort import natsorted
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from pipeline_code.PerformanceEvaluation import evaluate_model
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def apply_min_duration_filter(preds, min_duration=5, background_class="background"):
    preds = np.array(preds, dtype=object)
    i = 0
    while i < len(preds):
        cls = preds[i]
        if cls == background_class:
            i += 1
            continue
        j = i
        while j < len(preds) and preds[j] == cls:
            j += 1
        run_length = j - i
        if run_length < min_duration:
            replacement = preds[i - 1] if i > 0 else background_class
            preds[i:j] = replacement
            if replacement == cls:
                i = j
        else:
            i = j
    return preds


def apply_gap_fill(preds, max_gap=5, background_class="background"):
    preds = np.array(preds, dtype=object)
    i = 0
    while i < len(preds):
        if preds[i] != background_class:
            i += 1
            continue
        j = i
        while j < len(preds) and preds[j] == background_class:
            j += 1
        gap_length = j - i
        if gap_length <= max_gap and i > 0 and j < len(preds):
            before = preds[i - 1]
            after  = preds[j]
            if before == after and before != background_class:
                preds[i:j] = before
        i = j
    return preds


start = time.time()

# Define dataset version (e.g., "actual", "actual_1", "actual_2")
DATASET_VERSION = "hgb_3D_v4"

X_path = f"./pipeline_saved_processes/dataframes/X_3D.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_3D_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_3D.csv"
model_path = f"pipeline_saved_processes/models/HGB_{DATASET_VERSION}.pkl"

# checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    X_PATH = f"./pipeline_saved_processes/dataframes/X_3D.csv"
    Y_PATH = f"./pipeline_saved_processes/dataframes/y_3D.csv"
    COLLECTION_PATH="./pipeline_inputs/collection"
    LABELS_PATH = "./pipeline_inputs/labels"

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

    X = pd.read_csv(X_path, index_col=["video_id", "frame"])
    y = pd.read_csv(y_path, index_col=["video_id", "frame"])

# Apply pure collinearity filtering (no target variable used)
if os.path.isfile(X_filtered_path):
    print("Loading filtered X...")
    X = pd.read_csv(X_filtered_path, index_col=["video_id", "frame"])
else:
    print("Applying collinearity filter...")
    X = collinearity_filter(X, threshold=0.95)
    print("Saving filtered X...")
    X.to_csv(X_filtered_path)
    print("Filtered X saved!")

# OFT dataset: 20 videos (collection 1-21, no 5) — 14 train / 3 val / 3 test (70/15/15)
_val_ids  = [7, 14, 20]
_test_ids = [4, 11, 17]

if not os.path.isfile(model_path):

    # Split data using explicit OFT video IDs
    _all_ids    = list(X.index.get_level_values("video_id").unique())
    _held_out   = set(_val_ids + _test_ids)
    train_video_ids = [v for v in _all_ids if v not in _held_out]
    val_video_ids   = [v for v in _val_ids  if v in _all_ids]
    test_video_ids  = [v for v in _test_ids if v in _all_ids]
    print(f"Split: Train={len(train_video_ids)}, Val={len(val_video_ids)}, Test={len(test_video_ids)} videos")

    X_train = X.loc[X.index.get_level_values('video_id').isin(train_video_ids)]
    X_val   = X.loc[X.index.get_level_values('video_id').isin(val_video_ids)]
    X_test  = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
    y_train = y.loc[y.index.get_level_values('video_id').isin(train_video_ids)]
    y_val   = y.loc[y.index.get_level_values('video_id').isin(val_video_ids)]
    y_test  = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]

    # Get video groups for cross-validation
    groups_train = X_train.index.get_level_values("video_id")

    # Ravel
    y_train = y_train.values.ravel()
    y_val   = y_val.values.ravel()
    y_test  = y_test.values.ravel()

    # Calculate class weights for multi-class imbalanced data
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Class distribution in training: {class_counts}")

    # For multi-class, calculate sample weights
    total_samples = len(y_train)
    n_classes = len(unique)
    class_weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])
    print(f"Class weights: {class_weights}")

    # Create pipeline with preprocessing steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(random_state=42, early_stopping=True, verbose=0))
    ])

    # Grid Search
    param_grid = {
        'classifier__max_iter': [125],
        'classifier__max_depth': [4],
        'classifier__learning_rate': [0.1],
        'classifier__min_samples_leaf': [60], #the higher, the less overfitting, 80
        'classifier__l2_regularization': [0.00],
        'classifier__max_bins': [255],
        'classifier__max_leaf_nodes': [31]  # Limits tree complexity, 63

    }

    # Use GroupKFold for video-level cross-validation (4 folds with 15 train videos)
    cv_splitter = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splitter,
        scoring='f1_macro',
        n_jobs=2,
        verbose=2
    )

    grid_search.fit(X_train, y_train, groups=groups_train, classifier__sample_weight=sample_weights)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    evaluate_model(model, X_train, y_train, X_test, y_test, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")

    y_pred_test = model.predict(X_test)
    pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test}, index=X_test.index)
    all_true, all_smoothed = [], []
    for vid in sorted(pred_df.index.get_level_values('video_id').unique()):
        vid_df = pred_df.loc[vid]
        smoothed = apply_min_duration_filter(vid_df['y_pred'].values)
        smoothed = apply_gap_fill(smoothed)
        all_true.extend(vid_df['y_true'].values)
        all_smoothed.extend(smoothed)
    all_true = np.array(all_true)
    all_smoothed = np.array(all_smoothed)
    print(f"\nTest macro F1 (smoothed): {f1_score(all_true, all_smoothed, average='macro', zero_division=0):.4f}")
    print(classification_report(all_true, all_smoothed, zero_division=0))

    # Save model, class weights, and best parameters
    Shelf(X_train, X_test, model, model_path, model_weights=class_weights, best_params=best_params)

else:
    X_train, X_test, y_train, y_test, model, extra = Shelf.load(X, y, model_path, return_extra=True)
    class_weights = extra.get('model_weights', extra)
    best_params = extra.get('best_params', {})
    print(f"Loaded class weights: {class_weights}")
    print(f"Loaded best parameters: {best_params}")

    # Ensure y_train and y_test are raveled for evaluation
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

    # Reconstruct val set from known OFT IDs
    _all_ids = list(X.index.get_level_values("video_id").unique())
    val_video_ids = [v for v in _val_ids if v in _all_ids]
    X_val = X.loc[X.index.get_level_values('video_id').isin(val_video_ids)]
    y_val = y.loc[y.index.get_level_values('video_id').isin(val_video_ids)].values.ravel()

    # Print performance evaluation for loaded model
    print("\n=== Performance Evaluation for Loaded Model ===")
    evaluate_model(model, X_train, y_train, X_test, y_test, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")

    y_pred_test = model.predict(X_test)
    pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test}, index=X_test.index)
    all_true, all_smoothed = [], []
    for vid in sorted(pred_df.index.get_level_values('video_id').unique()):
        vid_df = pred_df.loc[vid]
        smoothed = apply_min_duration_filter(vid_df['y_pred'].values)
        smoothed = apply_gap_fill(smoothed)
        all_true.extend(vid_df['y_true'].values)
        all_smoothed.extend(smoothed)
    all_true = np.array(all_true)
    all_smoothed = np.array(all_smoothed)
    print(f"\nTest macro F1 (smoothed): {f1_score(all_true, all_smoothed, average='macro', zero_division=0):.4f}")
    print(classification_report(all_true, all_smoothed, zero_division=0))


# Ensure y_train and y_test are raveled for both branches
if not isinstance(y_train, np.ndarray):
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

# Calculate sample weights (for both new training and loaded model)
if 'sample_weights' not in locals():
    sample_weights = np.array([class_weights[y] for y in y_train])


# Save one-hot encoded predictions per video (same format as CNN_Transformer_OFT_predict.py)
PREDICTION_OUTPUT_FOLDER = "./pipeline_outputs/predictions_hgb_3D"
os.makedirs(PREDICTION_OUTPUT_FOLDER, exist_ok=True)

BEHAVIOR_NAMES = ["background", "Supportedrearing", "Unsupportedrearing", "Grooming"]
CLASS_TO_COLUMN = {
    "background":      "background",
    "supportedrear":   "Supportedrearing",
    "unsupportedrear": "Unsupportedrearing",
    "grooming":        "Grooming",
}

pred_df_save = pd.DataFrame({'y_pred': y_pred_test}, index=X_test.index)

for vid in sorted(pred_df_save.index.get_level_values('video_id').unique()):
    vid_df = pred_df_save.loc[vid]
    preds = vid_df['y_pred'].values
    frame_indices = vid_df.index.tolist()

    one_hot = np.zeros((len(preds), len(BEHAVIOR_NAMES)), dtype=int)
    for i, cls_name in enumerate(preds):
        col = CLASS_TO_COLUMN.get(cls_name, cls_name)
        if col in BEHAVIOR_NAMES:
            one_hot[i, BEHAVIOR_NAMES.index(col)] = 1

    df = pd.DataFrame(one_hot, columns=BEHAVIOR_NAMES)
    df.index.name = "frame"

    out_path = os.path.join(PREDICTION_OUTPUT_FOLDER, f"{vid}.csv")
    df.to_csv(out_path)
    print(f"Saved: {out_path}  ({len(df)} frames)")

elapsed = time.time() - start
print(f"\nDone. Predictions saved to: {PREDICTION_OUTPUT_FOLDER}")
print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
