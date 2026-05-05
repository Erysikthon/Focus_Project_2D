from pipeline_code.generate_features_ai import triangulate
from pipeline_code.generate_features_ai import features
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

# Define dataset version (e.g., "actual", "actual_1", "actual_2")
DATASET_VERSION = "hgb_3D_v2"

X_path = f"./pipeline_saved_processes/dataframes/X_3D.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_3D_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_3D.csv"
model_path = f"pipeline_saved_processes/models/HGB_{DATASET_VERSION}.pkl"

# checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    features_collection = triangulate(
        collection_path="./pipeline_inputs/collection",
        fps=30,

        rescale_points=("tr", "tl"),
        rescale_distance=0.64,
        filter_threshold=0.9,
        construction_points={"mid": {"between_points": ("tl", "tr", "bl", "br"), "mouse_or_oft": "oft"}, },
        smoothing=True,
        smoothing_mouse=3,
        smoothing_oft=20
    )

    X: pd.DataFrame = features(features_collection,

                               distance={("neck", "earl"),
                                         ("neck", "earr"),
                                         ("neck", "bcl"),
                                         ("neck", "bcr"),
                                         ("bcl", "hipl"),
                                         ("bcr", "hipr"),
                                         ("hipl", "tailbase"),
                                         ("hipr", "tailbase"),
                                         ("headcentre", "neck"),
                                         ("neck", "bodycentre"),
                                         ("bodycentre", "tailbase"),
                                         ("headcentre", "earl"),
                                         ("headcentre", "earr"),
                                         ("bodycentre", "bcl"),
                                         ("bodycentre", "bcr"),
                                         ("bodycentre", "hipl"),
                                         ("bodycentre", "hipr")
                                        },

                               height_diff = {("headcentre", "mid"),
                                              ("earl", "mid"),
                                              ("earr", "mid"),
                                              ("neck", "mid"),
                                              ("bcl", "mid"),
                                              ("bcr", "mid"),
                                              ("bodycentre", "mid"),
                                              ("hipl", "mid"),
                                              ("hipr", "mid"),
                                              ("tailcentre", "mid")
                                              },

                               # syntax for azimuth deviation: basepoint, line1, line2. ideally keep clockwise/counterclockwise
                               # notation consistent across function calls

                               # problem: both azimuth and azimuth deviation compute relative to x axis

                               # also sin and cos will have to be computed elsehow
                               angle={("neck", "bodycentre" "headcentre"),
                                      ("neck", "bodycentre", "earl"),
                                      ("neck", "bodycentre", "earr"),
                                      ("bodycentre", "tailbase", "neck"),
                                      ("tailbase", "bodycentre" "hipl"),
                                      ("tailbase", "bodycentre", "hipr"),
                                      #("tailbase", "bodycentre", "hipl", "bcl"),
                                      #("tailbase", "bodycentre", "hipr", "bcr"),
                                      ("tailbase", "bodycentre", "tailcentre"),
                                      #("bodycentre", "tailbase", "tailcentre", "tailtip")
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

                               volume={
                                   ("neck", "bodycentre", "bcl", "bcr"): ((0, 1, 2), (2, 1, 3), (0, 3, 1), (0, 2, 3)),
                                   ("bodycentre", "hipl", "tailbase", "hipr"): ((0, 3, 2), (3, 1, 2), (0, 2, 1),
                                                                                (0, 1, 3)),
                                   ("neck", "bcl", "hipl", "bodycentre"): ((0, 1, 3), (1, 2, 3), (3, 2, 0), (0, 2, 1)),
                                   ("neck", "bcr", "hipr", "bodycentre"): ((0, 3, 1), (1, 3, 2), (3, 0, 2), (0, 1, 2))
                                   },

                               standard_deviation=("headcentre.z",
                                                   "earl.z",
                                                   "earr.z",
                                                   "bodycentre.z",
                                                   "Volume_of_neck_bodycentre_bcl_bcr",
                                                   "Volume_of_bodycentre_hipl_tailbase_hipr",
                                                   "Volume_of_neck_bcl_hipl_bodycentre",
                                                   "Volume_of_neck_bcr_hipr_bodycentre"
                                                   ),

                               f_b_fill=True,

                               embedding_length=list(range(-15, 16, 3))
                               )

    y = labels(labels_path="./pipeline_inputs/labels",
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

    # Reconstruct val set from known OFT IDs
    _all_ids = list(X.index.get_level_values("video_id").unique())
    val_video_ids = [v for v in _val_ids if v in _all_ids]
    X_val = X.loc[X.index.get_level_values('video_id').isin(val_video_ids)]
    y_val = y.loc[y.index.get_level_values('video_id').isin(val_video_ids)].values.ravel()

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
