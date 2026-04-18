#import py3r.behaviour as py3r
from pipeline_code.generate_features import features_2d
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.filter_and_preprocess import collinearity_filter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_code.Shelf import Shelf
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from pipeline_code.model_tools import predict_multiIndex
from sklearn.model_selection import GridSearchCV, PredefinedSplit
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
import io
import contextlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


start = time.time()

# Define dataset version
DATASET_VERSION = "HGB_v1"

X_path = f"./pipeline_saved_processes/dataframes/X.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y.csv"
model_path = f"pipeline_saved_processes/models/HGB_{DATASET_VERSION}.pkl"

# checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    # Load 2D tracking data (single camera, no triangulation)
    from py3r.behaviour.tracking.tracking import Tracking
    from py3r.behaviour.features.features_collection import FeaturesCollection
    from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    import glob

    collection_path = "./data/tracking"
    fps = 30
    rescale_points = ("tr", "tl")
    rescale_distance_mbt = 0.47  # For MBT videos (27.5 x 37.5 cm box)
    rescale_distance_default = 0.64  # For other videos (45 x 45 cm box)
    filter_threshold = 0.9
    construction_points = {"mid": {"between_points": ("tl", "tr", "bl", "br"), "mouse_or_oft": "oft"}}
    smoothing = True
    smoothing_mouse = 3
    smoothing_oft = 20



    # Load tracking point CSVs from collection folder
    tracking_dict = {}
    csv_files = natsorted([f for f in os.listdir(collection_path) if f.endswith('.csv') and not f.startswith('.')])

    for csv_file in csv_files:
        video_handle = os.path.splitext(csv_file)[0]  # Use filename without extension as handle
        csv_path = os.path.join(collection_path, csv_file)
        tracking_dict[video_handle] = Tracking.from_yolo3r(filepath=csv_path, handle=video_handle, fps=fps)

    tracking_collection = TrackingCollection(tracking_dict)
    print(f"Initial videos loaded: {len(tracking_collection._obj_dict)}")

    # Strip column name prefixes (e.g., oft.oft_0.tr.x -> tr.x)
    tracking_collection.each.strip_column_names()

    # Filter out videos that don't have OFT corner tracking
    videos_to_remove = []
    for video_id, tracking in tracking_collection._obj_dict.items():
        required_columns = ['tr.x', 'tr.y', 'tl.x', 'tl.y', 'br.x', 'br.y', 'bl.x', 'bl.y']
        if not all(col in tracking.data.columns for col in required_columns):
            videos_to_remove.append(video_id)
            print(f"Warning: Video {video_id} missing OFT corner data - will be excluded")
            print(f"  Available columns: {[col for col in tracking.data.columns if any(x in col for x in ['tr', 'tl', 'br', 'bl'])]}")

    for video_id in videos_to_remove:
        del tracking_collection._obj_dict[video_id]

    print(f"After OFT filter: {len(tracking_collection._obj_dict)} videos with valid OFT tracking")

    # Likelihood filter
    tracking_collection.each.filter_likelihood(filter_threshold)

    # Rescale (2D only - x, y) with different distances based on video name
    for video_id, tracking in tracking_collection._obj_dict.items():
        if "MBT" in video_id:
            tracking.rescale_by_known_distance(rescale_points[0], rescale_points[1], rescale_distance_mbt, dims=("x", "y"))
            print(f"Rescaled {video_id} with distance {rescale_distance_mbt} (MBT)")
        else:
            tracking.rescale_by_known_distance(rescale_points[0], rescale_points[1], rescale_distance_default, dims=("x", "y"))
            print(f"Rescaled {video_id} with distance {rescale_distance_default} (default)")


    # Smoothing
    if smoothing:
        tracking_collection.each.smooth_all(3)


    features_collection = FeaturesCollection.from_tracking_collection(tracking_collection)

    X: pd.DataFrame = features_2d(features_collection,

                               distance={("neck", "earl"): ("x", "y"),
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
                                  # these next ones are angles
                                  # please note the syntax
                                  # first point is the "peak" of the angle, next 2 are the points defining the two lines
                                  # ALWAYS denote the directions in counterclockwise order, this function gives a signed result
                                  # failing to follow the above instructions will result in random signage (+/-)
                                  # refer to mouse tracking model for the points and directions
                               azimuth_deviation={("neck", "bcl", "earl"),
                                      ("neck", "earr", "bcr"),
                                      ("bodycentre", "hipl", "bcl"),
                                      ("bodycentre", "bcr", "hipr"),
                                      ("bodycentre", "tailbase", "hipl"),
                                      ("bodycentre", "hipr", "tailbase")
                               },
                                # just examples, idek if this feature is useful
                               azimuth={("bodycentre", "neck"),
                                      ("neck", "earl"),
                                      ("neck", "earr"),
                                      ("tailbase", "bodycentre"),
                                      ("tailbase", "hipr"),
                                      ("tailbase", "hipl"),
                                      ("neck", "nose")
                                      },

                               speed=("headcentre",
                                      "earl",
                                      "earr",
                                      "neck",
                                      "bcl",
                                      "bcr",
                                      "bodycentre",
                                      "hipl",
                                      "hipr",
                                      "tailcentre"
                                      ),

                               distance_change=("headcentre",
                                                "earl",
                                                "earr",
                                                "neck",
                                                "bcl",
                                                "bcr",
                                                "bodycentre",
                                                "hipl",
                                                "hipr",
                                                "tailcentre"
                               ),
                                # areas of the head and body, for now
                                # we can do individual triangles too
                                # write them "along the edge", otherwise the function gets confused
                               area_of_boundary={("nose", "earl", "neck", "earr"),
                                                 ("neck", "bcr", "hipr", "tailbase", "hipl", "bcl")
                               },

                               distance_to_boundary=("headcentre",
                                                     "earl",
                                                     "earr",
                                                     "neck",
                                                     "bcl",
                                                     "bcr",
                                                     "bodycentre",
                                                     "hipl",
                                                     "hipr",
                                                     "tailcentre"
                                                     ),


                               f_b_fill=True,

                               embedding_length=list(range(-15, 16, 1))
                               )

    y = labels(labels_path="./data/labels",
               )

    print(f"\nBefore drop_non_analyzed_videos: X has {X.index.get_level_values('video_id').nunique()} videos, y has {y.index.get_level_values('video_id').nunique()} videos")
    X, y = drop_non_analyzed_videos(X=X, y=y)
    print(f"After drop_non_analyzed_videos: {X.index.get_level_values('video_id').nunique()} videos")

    X, y = drop_last_frame(X=X, y=y)
    print(f"After drop_last_frame: {X.index.get_level_values('video_id').nunique()} videos")

    X, y = drop_nas(X=X, y=y)
    print(f"After drop_nas: {X.index.get_level_values('video_id').nunique()} videos")

    X = reduce_bits(X)

    print("saving...")
    X.to_csv(X_path)
    y.to_csv(y_path)
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

if not os.path.isfile(model_path):

    train_video_ids = ['20231123_10min_OFT-BL_3961', '20231123_10min_OFT-BL_3962', '20231123_10min_OFT-BL_3963', '20231123_10min_OFT-BL_3964', '20231123_10min_OFT-BL_4028', '3278_21min_behaviour_2023-01-19T11_08_30', 'BehavioralCamera2023-02-14T13_05_19_shorter', 'BehavioralCamera2023-02-14T15_22_37_shorter', 'BehavioralCamera2023-02-15T14_40_46_shorter', 'BehavioralCamera2023-02-18T10_33_06_shorter', 'BehavioralCamera2023-02-18T12_37_43_shorter', 'BehavioralCamera2023-02-23T15_42_37_shorter', 'BehavioralCamera2023-03-09T10_37_32', 'BehavioralCamera2023-03-09T11_04_40', 'BehavioralCamera2023-03-09T11_41_07', 'BehavioralCamera2023-03-09T12_34_50', 'BehavioralCamera2023-03-09T13_02_04', 'MBT1-M10', 'MBT1-M11', 'MBT1-M15', 'MBT1-M18', 'MBT1-M2', 'MBT1-M6', 'T1', 'T12', 'T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T2', 'T5', 'T8', 'T9']
    val_video_ids   = ['20231123_10min_OFT-BL_3919', '20231123_10min_OFT-BL_4029', 'BehavioralCamera2023-02-19T14_53_53_shorter', 'BehavioralCamera2023-03-09T14_30_45', 'MBT1-M14', 'MBT1-M3', 'T10', 'T3', 'T7']
    test_video_ids  = ['20231123_10min_OFT-BL_4025', '3279_21min_behaviour_2023-01-19T12_57_29', 'BehavioralCamera2023-02-23T10_23_42_shorter', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'BehavioralCamera2023-03-09T12_08_14', 'MBT1-M7', 'T11', 'T15', 'T4', 'T6']

    all_video_ids = X.index.get_level_values("video_id").unique()
    train_video_ids = [vid for vid in train_video_ids if vid in all_video_ids]
    val_video_ids   = [vid for vid in val_video_ids   if vid in all_video_ids]
    test_video_ids  = [vid for vid in test_video_ids  if vid in all_video_ids]

    print(f"Split: Train={len(train_video_ids)}, Val={len(val_video_ids)}, Test={len(test_video_ids)} videos")

    X_train = X.loc[X.index.get_level_values('video_id').isin(train_video_ids)]
    X_val   = X.loc[X.index.get_level_values('video_id').isin(val_video_ids)]
    X_test  = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
    y_train = y.loc[y.index.get_level_values('video_id').isin(train_video_ids)]
    y_val   = y.loc[y.index.get_level_values('video_id').isin(val_video_ids)]
    y_test  = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]

    # Reset index to avoid sklearn indexing issues with MultiIndex
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)

    # Convert to numpy and ravel (handle PyArrow types)
    y_train = np.array(y_train.values).ravel()
    y_val   = np.array(y_val.values).ravel()
    y_test  = np.array(y_test.values).ravel()

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

    # Compute sample weights for val set using the same class weights
    sample_weights_val = np.array([class_weights[y] for y in y_val])

    # Combine train and val for grid search with predefined split
    X_trainval = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_trainval = np.concatenate([y_train, y_val])
    sample_weights_combined = np.concatenate([sample_weights, sample_weights_val])

    # PredefinedSplit: -1 = train samples, 0 = val samples
    test_fold = np.concatenate([-np.ones(len(y_train), dtype=int), np.zeros(len(y_val), dtype=int)])
    ps = PredefinedSplit(test_fold)

    # Create pipeline with preprocessing steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(random_state=42, early_stopping=True, verbose=0))
    ])

    # Grid Search
    param_grid = {
        'classifier__max_iter': [300],
        'classifier__max_depth': [6, 7],  # Added 6 for shallower trees
        'classifier__learning_rate': [0.1],  # Added 0.08 for slower, better training
        'classifier__min_samples_leaf': [40, 50],  # Added 50, 60 to reduce overfitting
        'classifier__l2_regularization': [0, 0.1],  # Added 0.5 for stronger regularization
        'classifier__max_bins': [255],
        'classifier__max_leaf_nodes': [50, 63]  # Added 50 for fewer leaf nodes

    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=ps,
        scoring='f1_macro',
        n_jobs=2,
        verbose=2
    )

    grid_search.fit(X_trainval, y_trainval, classifier__sample_weight=sample_weights_combined)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    results_path = f"pipeline_outputs/results_{DATASET_VERSION}.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Dataset version: {DATASET_VERSION}")
        print(f"\nBest parameters:\n{json.dumps(best_params, indent=2)}")
        print(f"\nClass weights:\n{json.dumps({str(k): v for k, v in class_weights.items()}, indent=2)}")
        print("\n--- Test set evaluation ---")
        evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=10, conf_matrix_path=f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")
        print("\n--- Val set evaluation ---")
        evaluate_model(model, X_train, y_train, X_val, y_val, min_frames=10, conf_matrix_path=f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_val.png")

    results_text = buf.getvalue()
    print(results_text)
    with open(results_path, 'w') as f:
        f.write(results_text)
    print(f"Results saved to {results_path}")

    # Save model, class weights, and best parameters
    Shelf(X_train, X_test, model, model_path, model_weights=class_weights, best_params=best_params, X_val=X_val, y_val=y_val)

else:
    X_train, X_test, y_train, y_test, model, extra = Shelf.load(X, y, model_path, return_extra=True)
    class_weights = extra.get('model_weights', extra)
    best_params = extra.get('best_params', {})
    X_val = extra.get('X_val', None)
    y_val = extra.get('y_val', None)
    print(f"Loaded class weights: {class_weights}")
    print(f"Loaded best parameters: {best_params}")

    # Ensure y_train and y_test are raveled for evaluation
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
    if y_val is not None and not isinstance(y_val, np.ndarray):
        y_val = y_val.values.ravel()

    # Print performance evaluation for loaded model
    print("\n=== Performance Evaluation for Loaded Model ===")
    print("\nWith smoothing")
    evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")

    if X_val is not None:
        print("\nVal set evaluation")
        evaluate_model(model, X_train, y_train, X_val, y_val, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_val.png")
