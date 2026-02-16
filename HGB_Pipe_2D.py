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


start = time.time()

# Define dataset version
DATASET_VERSION = "2D_attempt_hist_1"

X_path = f"./pipeline_saved_processes/dataframes/X_2D_hist.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_2D_hist_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_2D_hist.csv"
model_path = f"pipeline_saved_processes/models/HGB_{DATASET_VERSION}.pkl"

# checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    # Load 2D tracking data (single camera, no triangulation)
    from py3r.behaviour.tracking.tracking import LoadOptions as opt, Tracking
    from py3r.behaviour.features.features_collection import FeaturesCollection
    from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    import glob

    collection_path = "./pipeline_inputs/collection_2D"
    fps = 30
    rescale_points = ("tr", "tl")
    rescale_distance = 0.64
    filter_threshold = 0.9
    construction_points = {"mid": {"between_points": ("tl", "tr", "bl", "br"), "mouse_or_oft": "oft"}}
    smoothing = True
    smoothing_mouse = 3
    smoothing_oft = 20

    options = opt(fps=fps)

    # Manually load each video's tracking from subfolders
    tracking_dict = {}
    video_folders = natsorted([d for d in os.listdir(collection_path) if os.path.isdir(os.path.join(collection_path, d))])

    for video_handle in video_folders:
        video_path = os.path.join(collection_path, video_handle)
        # Find the CSV file (e.g., T1.csv)
        csv_files = [f for f in os.listdir(video_path) if f.endswith('.csv') and not f.startswith('.')]
        if csv_files:
            csv_path = os.path.join(video_path, csv_files[0])
            tracking_dict[video_handle] = Tracking.from_yolo3r(filepath=csv_path, handle=video_handle, options=options)

    tracking_collection = TrackingCollection(tracking_dict)

    # Strip column name prefixes (e.g., oft.oft_0.tr.x -> tr.x)
    tracking_collection.strip_column_names()

    # Filter out videos that don't have OFT corner tracking
    videos_to_remove = []
    for video_id, tracking in tracking_collection._obj_dict.items():
        required_columns = ['tr.x', 'tr.y', 'tl.x', 'tl.y', 'br.x', 'br.y', 'bl.x', 'bl.y']
        if not all(col in tracking.data.columns for col in required_columns):
            videos_to_remove.append(video_id)
            print(f"Warning: Video {video_id} missing OFT corner data - will be excluded")

    for video_id in videos_to_remove:
        del tracking_collection._obj_dict[video_id]

    print(f"Loaded {len(tracking_collection._obj_dict)} videos with valid OFT tracking")

    # Likelihood filter
    tracking_collection.filter_likelihood(filter_threshold)

    # Rescale (2D only - x, y)
    tracking_collection.rescale_by_known_distance(rescale_points[0], rescale_points[1], rescale_distance, dims=("x", "y"))

    # Construction points
    if construction_points:
        for handle in construction_points:
            construction_infos = construction_points[handle]
            tracking_collection.construction_point(handle, construction_infos["between_points"], dims=("x", "y"))

    # Smoothing
    if smoothing:
        smoothing_dict = {
            # mouse
            "nose": {"window": smoothing_mouse, "type": "mean"},
            "headcentre": {"window": smoothing_mouse, "type": "mean"},
            "neck": {"window": smoothing_mouse, "type": "mean"},
            "earl": {"window": smoothing_mouse, "type": "mean"},
            "earr": {"window": smoothing_mouse, "type": "mean"},
            "bodycentre": {"window": smoothing_mouse, "type": "mean"},
            "bcl": {"window": smoothing_mouse, "type": "mean"},
            "bcr": {"window": smoothing_mouse, "type": "mean"},
            "hipl": {"window": smoothing_mouse, "type": "mean"},
            "hipr": {"window": smoothing_mouse, "type": "mean"},
            "tailbase": {"window": smoothing_mouse, "type": "mean"},
            "tailcentre": {"window": smoothing_mouse, "type": "mean"},
            "tailtip": {"window": smoothing_mouse, "type": "mean"},
            # oft
            "tr": {"window": smoothing_oft, "type": "median"},
            "tl": {"window": smoothing_oft, "type": "median"},
            "br": {"window": smoothing_oft, "type": "median"},
            "bl": {"window": smoothing_oft, "type": "median"},
            "mid": {"window": smoothing_oft, "type": "median"}
        }
        tracking_collection.smooth(smoothing_dict)

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

                               angle={("bodycentre", "neck", "neck", "headcentre"): "radians",
                                      ("bodycentre", "neck", "neck", "earl"): "radians",
                                      ("bodycentre", "neck", "neck", "earr"): "radians",
                                      ("tailbase", "bodycentre", "bodycentre", "neck"): "radians",
                                      ("tailbase", "bodycentre", "tailbase", "hipl"): "radians",
                                      ("tailbase", "bodycentre", "tailbase", "hipr"): "radians",
                                      ("tailbase", "bodycentre", "hipl", "bcl"): "radians",
                                      ("tailbase", "bodycentre", "hipr", "bcr"): "radians",
                                      ("bodycentre", "tailbase", "tailbase", "tailcentre"): "radians",
                                      ("bodycentre", "tailbase", "tailcentre", "tailtip"): "radians"
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

                               is_point_recognized=(["nose"]),

                               f_b_fill=True,

                               embedding_length=list(range(-15, 16, 3))
                               )

    y = labels(labels_path="./pipeline_inputs/labels_2D",
               )

    X, y = drop_non_analyzed_videos(X=X, y=y)
    X, y = drop_last_frame(X=X, y=y)
    X, y = drop_nas(X=X, y=y)
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

    # Split data (collinearity filtering already applied)
    X_train, X_test, y_train, y_test = video_train_test_split(X, y, test_videos=20, random_state =20)

    # Get video groups for cross-validation
    groups_train = X_train.index.get_level_values("video_id")

    # Ravel
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

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

    print("\nWith smoothing")
    evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")


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

    # Print performance evaluation for loaded model
    print("\n=== Performance Evaluation for Loaded Model ===")
    print("\nWith smoothing")
    evaluate_model(model, X_train, y_train, X_test, y_test, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_1.png")


# Ensure y_train and y_test are raveled for both branches
if not isinstance(y_train, np.ndarray):
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

# Calculate sample weights (for both new training and loaded model)
if 'sample_weights' not in locals():
    sample_weights = np.array([class_weights[y] for y in y_train])


# Extract feature importances using  permutation_importance
feature_importance_path = f'./pipeline_saved_processes/selected_features/HGB_{DATASET_VERSION}_selected_features.csv'


# Permutation Importance
if os.path.isfile(feature_importance_path):
    print("Loading existing permutation importance...")
    feature_importance_df = pd.read_csv(feature_importance_path)
    print(f"Features with importance > 0: {len(feature_importance_df)}")
    print(feature_importance_df.head(20))
else:
    print("Calculating permutation importance...")
    result = permutation_importance(
     model,
     X_train,
     y_train,
     n_repeats=5,
     random_state=42,
     n_jobs=2
    )
    importances = result.importances_mean
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Filter features with importance > 0.0001
    feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0.0001]
    print(f"Features with importance > 0.0001: {len(feature_importance_df)}")
    print(feature_importance_df.head(20))

    # Save selected features
    feature_importance_df.to_csv(f'./pipeline_saved_processes/selected_features/HGB_{DATASET_VERSION}_selected_features.csv', index=False)


# Plot top 300 feature importances
top_n_plot = 300
top_features_plot = feature_importance_df.head(top_n_plot)
plt.figure(figsize=(10, 12))
plt.barh(range(len(top_features_plot)), top_features_plot['Importance'], align='center')
plt.yticks(range(len(top_features_plot)), top_features_plot['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
model_name =  "Histogram Gradient Boosting"
plt.title(f'Top {top_n_plot} {model_name} Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'pipeline_outputs/feature_importances_HGB_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')
plt.close()

# Train second HGB model with only selected features
print("\nSecond HGB model with selected features...")
selected_features = feature_importance_df['Feature'].tolist()

HGB_selected_path = f"pipeline_saved_processes/models/HGB_{DATASET_VERSION}_selected_features.pkl"

if not os.path.isfile(HGB_selected_path):
    # Filter X to keep only selected features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Extract best hyperparameters from grid search (remove 'classifier__' prefix)
    best_clf_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    print(f"Using best parameters from grid search: {best_clf_params}")

    # Create pipeline with selected features using best hyperparameters
    print(f"Training HGB with {len(selected_features)} selected features...")
    pipeline_selected = Pipeline([
     ('scaler', StandardScaler()),
     ('classifier', HistGradientBoostingClassifier(
         random_state=42,
         early_stopping=False,
         verbose=0,
         **best_clf_params
     ))
    ])

    pipeline_selected.fit(X_train_sel, y_train, classifier__sample_weight=sample_weights)

    print("Evaluating model with selected features:")

    print("\nWith smoothing")
    evaluate_model(pipeline_selected, X_train_sel, y_train, X_test_sel, y_test, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_2.png")

    # Save the model
    Shelf(X_train_sel, X_test_sel, pipeline_selected, HGB_selected_path, model_weights=class_weights)

else:
    # Load the second model with selected features
    X_train_sel, X_test_sel, y_train_sel, y_test_sel, pipeline_selected, extra_sel = Shelf.load(X, y, HGB_selected_path, return_extra=True)

    # Ensure y arrays are raveled
    if not isinstance(y_train_sel, np.ndarray):
        y_train_sel = y_train_sel.values.ravel()
        y_test_sel = y_test_sel.values.ravel()

    print("\n=== Performance Evaluation for Loaded Second Model (Selected Features) ===")
    print("\nWith smoothing")
    evaluate_model(pipeline_selected, X_train_sel, y_train_sel, X_test_sel, y_test_sel, min_frames=10, conf_matrix_path = f"pipeline_outputs/conf_matrix_{DATASET_VERSION}_model_2.png")
