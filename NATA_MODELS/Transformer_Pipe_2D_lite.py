import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pipeline_code.generate_features import features_2d
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.model_tools import video_train_test_split
from pipeline_code.filter_and_preprocess import collinearity_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from natsort import natsorted
import os
import numpy as np
import math


start = time.time()

# Define dataset version
DATASET_VERSION = "Transformer"

X_path = f"./pipeline_saved_processes/dataframes/X_everything.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_everything_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_everything.csv"
model_path = f"pipeline_saved_processes/models/Transformer_{DATASET_VERSION}.pth"
scaler_path = f"pipeline_saved_processes/models/scaler_{DATASET_VERSION}.pkl"
label_encoder_path = f"pipeline_saved_processes/models/label_encoder_{DATASET_VERSION}.pkl"

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

    # Likelihood filter (before stripping column names)
    tracking_collection.each.filter_likelihood(filter_threshold)

    # Rescale (2D only - x, y) with different distances based on video name (before stripping column names)
    for video_id, tracking in tracking_collection._obj_dict.items():
        # Get available point names from column names
        # Columns are like: 'oft.oft_0.tr.x', 'oft.oft_0.tr.y', etc.
        columns = tracking.data.columns

        # Extract unique point names (everything before .x, .y, .z, .likelihood)
        point_names = set()
        for col in columns:
            parts = col.split('.')
            if len(parts) >= 2 and parts[-1] in ['x', 'y', 'z', 'likelihood']:
                point_name = '.'.join(parts[:-1])
                point_names.add(point_name)

        # Find points that end with 'tr' and 'tl'
        tr_point = next((p for p in point_names if p.endswith('.tr')), None)
        tl_point = next((p for p in point_names if p.endswith('.tl')), None)

        if tr_point and tl_point:
            if "MBT" in video_id:
                tracking.rescale_by_known_distance(tr_point, tl_point, rescale_distance_mbt, dims=("x", "y"))
                print(f"Rescaled {video_id} with distance {rescale_distance_mbt} (MBT) using {tr_point} and {tl_point}")
            else:
                tracking.rescale_by_known_distance(tr_point, tl_point, rescale_distance_default, dims=("x", "y"))
                print(f"Rescaled {video_id} with distance {rescale_distance_default} (default) using {tr_point} and {tl_point}")
        else:
            print(f"Warning: Could not find tr/tl points for {video_id}, skipping rescaling")

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

    # Smoothing
    if smoothing:
        tracking_collection.each.smooth_all(smoothing_mouse)

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

                               embedding_length=list(range(-15, 16, 1))
                               )

    y = labels(labels_path="./pipeline_inputs/labels",
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

# PyTorch Dataset class for sequences
class SequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length=30, stride=10):
        """
        Args:
            X: DataFrame with multi-index (video_id, frame)
            y: DataFrame with multi-index (video_id, frame)
            sequence_length: Number of frames in each sequence
            stride: Step size between sequences (default 10 to reduce memory)
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        self.sequence_info = []  # Track (video_id, start_frame) for consensus voting

        # Group by video_id
        for video_id in X.index.get_level_values('video_id').unique():
            video_X = X.loc[video_id].values
            video_y = y.loc[video_id].values.ravel()

            # Create sequences with stride to reduce memory usage
            for i in range(0, len(video_X) - sequence_length + 1, stride):
                seq = video_X[i:i + sequence_length]
                labels_seq = video_y[i:i + sequence_length]  # All frame labels for the sequence
                self.sequences.append(seq)
                self.labels.append(labels_seq)
                self.sequence_info.append((video_id, i))  # Store metadata

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.LongTensor(np.array(self.labels).astype(np.int64))
        print(f"Created {len(self.sequences)} sequences with per-frame labels (stride={stride})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dim_feedforward=512, dropout=0.3):
        super(TransformerClassifier, self).__init__()

        # Simple input projection (matches CNNTransformerClassifier style)
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simple classification head (matches CNNTransformerClassifier style)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)   # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)        # (batch, seq_len, d_model)
        return self.classifier(x)      # (batch, seq_len, num_classes)


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, seq_len, num_classes)

        # Reshape for loss computation
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(batch_size * seq_len, num_classes)
        labels_flat = batch_y.view(batch_size * seq_len)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler per batch for warmup+cosine schedule
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 2)  # (batch, seq_len)
        total += batch_y.numel()
        correct += (predicted == batch_y).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


# Evaluation function
def evaluate(model, dataloader, device, use_consensus=True):
    """
    Evaluate model with per-frame predictions

    Args:
        use_consensus: If True, uses majority voting across overlapping sequences for each unique frame
                      If False, treats all predictions independently (inflates metrics)
    """
    model.eval()

    if not use_consensus:
        # Original behavior: treat all predictions independently
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)  # (batch, seq_len, num_classes)
                _, predicted = torch.max(outputs, 2)  # (batch, seq_len)
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.numpy().flatten())
        return np.array(all_preds), np.array(all_labels)

    # Consensus voting: aggregate predictions per unique (video_id, frame_idx)
    from collections import defaultdict
    from scipy import stats

    frame_predictions = defaultdict(list)  # {(video_id, frame_idx): [pred1, pred2, ...]}
    frame_labels = {}  # {(video_id, frame_idx): true_label}

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)  # (batch, seq_len, num_classes)
            _, predicted = torch.max(outputs, 2)  # (batch, seq_len)

            predicted_np = predicted.cpu().numpy()
            labels_np = batch_y.numpy()

            # Get metadata for this batch
            batch_size = batch_X.shape[0]
            for b in range(batch_size):
                # Calculate which sequence this is in the dataset
                seq_idx = batch_idx * dataloader.batch_size + b
                if seq_idx >= len(dataloader.dataset):
                    continue

                video_id, start_frame = dataloader.dataset.sequence_info[seq_idx]

                # Record predictions for each frame in this sequence
                for frame_offset in range(dataloader.dataset.sequence_length):
                    frame_idx = start_frame + frame_offset
                    key = (video_id, frame_idx)

                    frame_predictions[key].append(predicted_np[b, frame_offset])
                    frame_labels[key] = labels_np[b, frame_offset]  # Same label from all sequences

    # Apply majority voting
    consensus_preds = []
    consensus_labels = []

    for key in sorted(frame_predictions.keys()):
        # Majority vote (mode)
        preds = frame_predictions[key]
        consensus_pred = stats.mode(preds, keepdims=False)[0]

        consensus_preds.append(consensus_pred)
        consensus_labels.append(frame_labels[key])

    return np.array(consensus_preds), np.array(consensus_labels)


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

    # ========== GRID SEARCH CONFIGURATION ==========
    # K-fold cross-validation on the TRAINING SET to find optimal hyperparameters.
    # Test set is never used during grid search - only for final evaluation!!!!

    ENABLE_GRID_SEARCH = True  # Set to False to skip grid search and use best params directly

    # Grid search configuration (32 combinations)
    param_grid = {
        'd_model': [256, 512],
        'num_layers': [3, 4],
        'nhead': [4, 8],
        'dim_feedforward': [1024],
        'dropout': [0.3],
        'lr': [0.0001, 0.0003],
        'batch_size': [512],
        'sequence_length': [30, 60],
        'stride': [5]
    }

    n_folds = 3  # K-fold cross-validation
    max_epochs_per_trial = 30  # Reduced epochs for grid search
    early_stop_patience = 10  # Early stopping patience for each trial
    # ===============================================

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = pd.DataFrame(
        label_encoder.fit_transform(y_train.values.ravel()),
        index=y_train.index,
        columns=[y_train.name] if isinstance(y_train, pd.Series) else y_train.columns
    )
    y_test_encoded = pd.DataFrame(
        label_encoder.transform(y_test.values.ravel()),
        index=y_test.index,
        columns=[y_test.name] if isinstance(y_test, pd.Series) else y_test.columns
    )
    y_val_encoded = pd.DataFrame(
        label_encoder.transform(y_val.values.ravel()),
        index=y_val.index,
        columns=[y_val.name] if isinstance(y_val, pd.Series) else y_val.columns
    )

    # Save label encoder
    import joblib
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Calculate class weights with stronger weighting for minority classes
    y_train_flat = y_train_encoded.values.ravel()
    unique, counts = np.unique(y_train_flat, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Class distribution in training: {class_counts}")

    total_samples = len(y_train_flat)
    n_classes = len(unique)
    num_classes = n_classes  # For compatibility with grid search code
    # Use stronger weighting (between sqrt and full inverse) via power of 0.7
    class_weights = {cls: (total_samples / (n_classes * count)) ** 0.7 for cls, count in class_counts.items()}
    print(f"Class weights (0.7 power scaled): {class_weights}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns
    )

    # Save scaler
    joblib.dump(scaler, scaler_path)

    # ========== GRID SEARCH WITH K-FOLD CV ==========
    if ENABLE_GRID_SEARCH:
        print("\n" + "="*60)
        print("STARTING GRID SEARCH WITH K-FOLD CROSS-VALIDATION")
        print("="*60)

        from itertools import product
        from sklearn.model_selection import KFold

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        print(f"\nTotal combinations to test: {len(all_combinations)}")
        print(f"K-folds: {n_folds}")
        print(f"Max epochs per trial: {max_epochs_per_trial}\n")

        # Get unique training video IDs for k-fold split
        train_video_ids_array = X_train.index.get_level_values('video_id').unique().to_numpy()
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        best_val_f1 = 0.0
        best_params = None
        results = []

        for combo_idx, param_combo in enumerate(all_combinations):
            params = dict(zip(param_names, param_combo))
            print(f"\n{'='*60}")
            print(f"Testing combination {combo_idx + 1}/{len(all_combinations)}")
            print(f"Parameters: {params}")
            print(f"{'='*60}")

            fold_f1_scores = []

            # K-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_video_ids_array)):
                print(f"\n--- Fold {fold + 1}/{n_folds} ---")

                # Split videos into train/val for this fold
                fold_train_videos = train_video_ids_array[train_idx]
                fold_val_videos = train_video_ids_array[val_idx]

                X_fold_train = X_train_scaled.loc[X_train_scaled.index.get_level_values('video_id').isin(fold_train_videos)]
                X_fold_val = X_train_scaled.loc[X_train_scaled.index.get_level_values('video_id').isin(fold_val_videos)]
                y_fold_train = y_train_encoded.loc[y_train_encoded.index.get_level_values('video_id').isin(fold_train_videos)]
                y_fold_val = y_train_encoded.loc[y_train_encoded.index.get_level_values('video_id').isin(fold_val_videos)]

                # Create datasets with current hyperparameters
                fold_train_dataset = SequenceDataset(X_fold_train, y_fold_train,
                                                    sequence_length=params['sequence_length'],
                                                    stride=params['stride'])
                fold_val_dataset = SequenceDataset(X_fold_val, y_fold_val,
                                                  sequence_length=params['sequence_length'],
                                                  stride=params['stride'])

                fold_train_loader = DataLoader(fold_train_dataset, batch_size=params['batch_size'],
                                              shuffle=True, num_workers=0, pin_memory=True)
                fold_val_loader = DataLoader(fold_val_dataset, batch_size=params['batch_size'],
                                            shuffle=False, num_workers=0, pin_memory=True)

                # Initialize model with current hyperparameters
                input_size = X_train.shape[1]
                device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

                fold_model = TransformerClassifier(
                    input_size=input_size,
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    num_layers=params['num_layers'],
                    num_classes=num_classes,
                    dim_feedforward=params['dim_feedforward'],
                    dropout=params['dropout']
                ).to(device)

                # Setup training
                weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)]).to(device)
                fold_criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
                fold_optimizer = optim.AdamW(fold_model.parameters(), lr=params['lr'],
                                            weight_decay=0.01, betas=(0.9, 0.999))

                # Simple learning rate schedule for quick training
                fold_scheduler = optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=max_epochs_per_trial)

                # Train for limited epochs
                best_fold_f1 = 0.0
                patience_counter = 0

                for epoch in range(max_epochs_per_trial):
                    train_loss, train_acc = train_epoch(fold_model, fold_train_loader, fold_criterion,
                                                       fold_optimizer, device, scheduler=None)
                    fold_scheduler.step()

                    # Evaluate on validation set
                    y_pred_val, y_true_val = evaluate(fold_model, fold_val_loader, device)
                    val_f1 = f1_score(y_true_val, y_pred_val, average='macro')

                    if val_f1 > best_fold_f1:
                        best_fold_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            print(f"  Early stopping at epoch {epoch + 1}")
                            break

                    if (epoch + 1) % 5 == 0:
                        print(f"  Epoch {epoch + 1}/{max_epochs_per_trial}: Val F1 = {val_f1:.4f}")

                print(f"  Best F1 for fold {fold + 1}: {best_fold_f1:.4f}")
                fold_f1_scores.append(best_fold_f1)

                # Clean up
                del fold_model, fold_train_loader, fold_val_loader, fold_train_dataset, fold_val_dataset
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Calculate mean F1 across folds
            mean_f1 = np.mean(fold_f1_scores)
            std_f1 = np.std(fold_f1_scores)

            print(f"\nMean validation F1: {mean_f1:.4f} (+/- {std_f1:.4f})")

            results.append({
                'params': params,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'fold_scores': fold_f1_scores
            })

            # Track best parameters
            if mean_f1 > best_val_f1:
                best_val_f1 = mean_f1
                best_params = params
                print(f"*** NEW BEST PARAMETERS! F1: {best_val_f1:.4f} ***")

        # Save grid search results
        import json
        results_path = f"pipeline_saved_processes/models/grid_search_results_{DATASET_VERSION}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_f1': best_val_f1,
                'all_results': [{'params': r['params'], 'mean_f1': r['mean_f1'], 'std_f1': r['std_f1']}
                               for r in results]
            }, f, indent=2)

        print("\n" + "="*60)
        print("GRID SEARCH COMPLETE")
        print("="*60)
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print(f"\nResults saved to: {results_path}")
        print("\nTop 3 configurations:")
        sorted_results = sorted(results, key=lambda x: x['mean_f1'], reverse=True)[:3]
        for i, r in enumerate(sorted_results):
            print(f"{i+1}. F1={r['mean_f1']:.4f} (+/-{r['std_f1']:.4f}): {r['params']}")

        # Use best parameters for final training
        SEQUENCE_LENGTH = best_params['sequence_length']
        STRIDE = best_params['stride']
        d_model = best_params['d_model']
        num_layers = best_params['num_layers']
        nhead = best_params['nhead']
        dim_feedforward = best_params['dim_feedforward']
        dropout = best_params['dropout']
        learning_rate = best_params['lr']
        batch_size = best_params['batch_size']
    else:
        # Use default parameters (or load from a previous grid search)
        SEQUENCE_LENGTH = 60
        STRIDE = 3
        d_model = 768
        nhead = 8
        num_layers = 6
        dim_feedforward = 3072
        dropout = 0.4
        learning_rate = 0.0003
        batch_size = 512
    # ================================================

    # Create sequence datasets with selected hyperparameters
    print(f"\nCreating final datasets with length {SEQUENCE_LENGTH} and stride {STRIDE}...")
    train_dataset = SequenceDataset(X_train_scaled, y_train_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    val_dataset   = SequenceDataset(X_val_scaled,   y_val_encoded,   sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    test_dataset  = SequenceDataset(X_test_scaled,  y_test_encoded,  sequence_length=SEQUENCE_LENGTH, stride=STRIDE)

    print(f"Total training sequences: {len(train_dataset)}, val sequences: {len(val_dataset)}, test sequences: {len(test_dataset)}")
    print(f"Class distribution in training sequences: {Counter(train_dataset.labels.numpy())}")

    # Use batch size from grid search or default
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model - prioritize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    input_size = X_train.shape[1]
    num_classes = len(unique)

    print(f"\n{'='*60}")
    print("FINAL MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"d_model: {d_model}")
    print(f"nhead: {nhead}")
    print(f"num_layers: {num_layers}")
    print(f"dim_feedforward: {dim_feedforward}")
    print(f"dropout: {dropout}")
    print(f"learning_rate: {learning_rate}")
    print(f"batch_size: {batch_size}")
    print(f"sequence_length: {SEQUENCE_LENGTH}")
    print(f"stride: {STRIDE}")
    print(f"{'='*60}\n")

    model = TransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CrossEntropyLoss with label smoothing for better generalization
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    # Optimizer with learning rate from grid search
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

    # Cosine annealing with warmup
    num_epochs = 150  # Reduced since cosine schedule will naturally decay
    warmup_epochs = 10
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_f1 = 0.0
    patience = 20  # Increased patience for larger model
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement to reset patience

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)

        # Evaluate on val set (for early stopping)
        y_pred_val, y_true_val = evaluate(model, val_loader, device)
        val_acc = 100 * np.sum(y_pred_val == y_true_val) / len(y_true_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

        # Early stopping with minimum delta
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'num_classes': num_classes,
                'sequence_length': SEQUENCE_LENGTH,
                'class_weights': class_weights,
                'train_videos': X_train.index.get_level_values('video_id').unique().tolist(),
                'val_videos': X_val.index.get_level_values('video_id').unique().tolist(),
                'test_videos': X_test.index.get_level_values('video_id').unique().tolist()
            }, model_path)
            print(f"  → New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

else:
    # Load existing model
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    input_size = checkpoint['input_size']
    d_model = checkpoint['d_model']
    nhead = checkpoint['nhead']
    num_layers = checkpoint['num_layers']
    dim_feedforward = checkpoint['dim_feedforward']
    num_classes = checkpoint['num_classes']
    SEQUENCE_LENGTH = checkpoint['sequence_length']
    class_weights = checkpoint['class_weights']

    model = TransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_feedforward,
        dropout=0.4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Recreate train/val/test split using saved video IDs
    train_video_ids = checkpoint['train_videos']
    val_video_ids   = checkpoint.get('val_videos', [])
    test_video_ids  = checkpoint['test_videos']

    X_train = X.loc[X.index.get_level_values('video_id').isin(train_video_ids)]
    X_val   = X.loc[X.index.get_level_values('video_id').isin(val_video_ids)]
    X_test  = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
    y_train = y.loc[y.index.get_level_values('video_id').isin(train_video_ids)]
    y_val   = y.loc[y.index.get_level_values('video_id').isin(val_video_ids)]
    y_test  = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]

    # Load label encoder and encode labels
    label_encoder = joblib.load(label_encoder_path)
    y_train_encoded = pd.DataFrame(
        label_encoder.transform(y_train.values.ravel()),
        index=y_train.index,
        columns=[y_train.name] if isinstance(y_train, pd.Series) else y_train.columns
    )
    y_val_encoded = pd.DataFrame(
        label_encoder.transform(y_val.values.ravel()),
        index=y_val.index,
        columns=[y_val.name] if isinstance(y_val, pd.Series) else y_val.columns
    )
    y_test_encoded = pd.DataFrame(
        label_encoder.transform(y_test.values.ravel()),
        index=y_test.index,
        columns=[y_test.name] if isinstance(y_test, pd.Series) else y_test.columns
    )

    # Load scaler and scale data
    scaler = joblib.load(scaler_path)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    # Create sequence datasets with stride
    STRIDE = 10  # Match training stride
    train_dataset = SequenceDataset(X_train_scaled, y_train_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    val_dataset   = SequenceDataset(X_val_scaled,   y_val_encoded,   sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    test_dataset  = SequenceDataset(X_test_scaled,  y_test_encoded,  sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

    print(f"Loaded model from {model_path}")
    print(f"Using device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")

# Training set evaluation
print("\n=== Training Set Evaluation ===")
y_pred_train, y_true_train = evaluate(model, train_loader, device)

train_acc = 100 * np.sum(y_pred_train == y_true_train) / len(y_true_train)
train_f1 = f1_score(y_true_train, y_pred_train, average='macro')

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Train F1 Score (macro): {train_f1:.4f}")
print("\nTraining Classification Report:")
print(classification_report(y_true_train, y_pred_train, target_names=label_encoder.classes_))

# Val set evaluation
print("\n=== Val Set Evaluation ===")
y_pred_val, y_true_val = evaluate(model, val_loader, device)

val_acc = 100 * np.sum(y_pred_val == y_true_val) / len(y_true_val)
val_f1 = f1_score(y_true_val, y_pred_val, average='macro')

print(f"Val Accuracy: {val_acc:.2f}%")
print(f"Val F1 Score (macro): {val_f1:.4f}")
print("\nVal Classification Report:")
print(classification_report(y_true_val, y_pred_val, target_names=label_encoder.classes_))

cm_val = confusion_matrix(y_true_val, y_pred_val)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix Val - {DATASET_VERSION}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'pipeline_outputs/conf_matrix_{DATASET_VERSION}_val.png', dpi=300, bbox_inches='tight')
plt.close()

# Test set evaluation
print("\n=== Test Set Evaluation ===")
y_pred, y_true = evaluate(model, test_loader, device)

# Metrics
test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
test_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test F1 Score (macro): {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {DATASET_VERSION}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'pipeline_outputs/conf_matrix_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nTotal time: {time.time() - start:.2f} seconds")
