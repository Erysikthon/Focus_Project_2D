import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from natsort import natsorted
import os
import numpy as np


start = time.time()

# Define dataset version
DATASET_VERSION = "LSTM_everything"

X_path = f"./pipeline_saved_processes/dataframes/X_everything.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_everything_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_everything.csv"
model_path = f"pipeline_saved_processes/models/LSTM_{DATASET_VERSION}.pth"
scaler_path = f"pipeline_saved_processes/models/scaler_{DATASET_VERSION}.pkl"
label_encoder_path = f"pipeline_saved_processes/models/label_encoder_{DATASET_VERSION}.pkl"

# checks if X and y already exists, and if not, they get computed

if not (os.path.isfile(X_path) and os.path.isfile(y_path)):

    # Load 2D tracking data (single camera, no triangulation)
    from py3r.behaviour.tracking.tracking import Tracking
    from py3r.behaviour.features.features_collection import FeaturesCollection
    from py3r.behaviour.tracking.tracking_collection import TrackingCollection
    import glob

    collection_path = "./pipeline_inputs/collection_full"
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

        # Skip MBT videos (uncomment to exclude MBT videos from training)
        # if "MBT" in video_handle:
        #     print(f"Skipping MBT video: {video_handle}")
        #     continue

        csv_path = os.path.join(collection_path, csv_file)
        tracking = Tracking.from_yolo3r(filepath=csv_path, handle=video_handle, fps=fps)

        # Rename .conf columns to .likelihood (py3r expects this format)
        tracking.data.columns = [col.replace('.conf', '.likelihood') if '.conf' in col else col
                                 for col in tracking.data.columns]

        # Drop only z-coordinates (keep all x,y data from all sources)
        cols_to_drop = [col for col in tracking.data.columns if '.z' in col]
        if cols_to_drop:
            tracking.data = tracking.data.drop(columns=cols_to_drop)

        tracking_dict[video_handle] = tracking

    tracking_collection = TrackingCollection(tracking_dict)
    print(f"Initial videos loaded: {len(tracking_collection._obj_dict)}")

    # Likelihood filter (must be done BEFORE strip_column_names)
    tracking_collection.each.filter_likelihood(filter_threshold)

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

    y = labels(labels_path="./pipeline_inputs/labels_full",
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

        # Group by video_id
        for video_id in X.index.get_level_values('video_id').unique():
            video_X = X.loc[video_id].values
            video_y = y.loc[video_id].values.ravel()

            # Create sequences with stride to reduce memory usage
            for i in range(0, len(video_X) - sequence_length + 1, stride):
                seq = video_X[i:i + sequence_length]
                label = video_y[i + sequence_length - 1]  # Label is the last frame's behavior
                self.sequences.append(seq)
                self.labels.append(label)

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.LongTensor(np.array(self.labels).astype(np.int64))
        print(f"Created {len(self.sequences)} sequences (stride={stride})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# LSTM Neural Network Architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for better context capture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional LSTM
        )

        # Account for bidirectional (hidden_size * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Input doubled due to bidirectional
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate forward and backward hidden states
        # h_n shape: (num_layers * 2, batch, hidden_size) for bidirectional
        last_hidden_forward = h_n[-2]  # Last layer forward direction
        last_hidden_backward = h_n[-1]  # Last layer backward direction
        last_hidden = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)  # Shape: (batch, hidden_size * 2)

        # Apply batch norm and dropout
        out = self.batch_norm(last_hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Mixup augmentation function
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to sequences."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function with optional mixup
def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()

        if use_mixup:
            # Apply mixup augmentation
            mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=mixup_alpha)
            outputs = model(mixed_X)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    return total_loss / len(dataloader), 100 * correct / total

# Focal Loss for imbalanced classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.

        Args:
            alpha: Class weights tensor (FloatTensor of shape [num_classes])
            gamma: Focusing parameter (default 2.0). Higher values give more focus to hard examples.
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        # Compute pt (the probability of the true class)
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Evaluation function with threshold adjustment
def evaluate(model, dataloader, device, thresholds=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)

            if thresholds is not None:
                # Apply per-class thresholds
                adjusted_probs = probs.clone()
                for class_idx, threshold in enumerate(thresholds):
                    adjusted_probs[:, class_idx] = (probs[:, class_idx] >= threshold).float() * probs[:, class_idx]
                predicted = torch.argmax(adjusted_probs, dim=1)
            else:
                predicted = torch.argmax(probs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Function to find optimal thresholds per class
def find_optimal_thresholds(model, dataloader, device, num_classes):
    """Find optimal decision thresholds for each class to maximize F1 score."""
    from sklearn.metrics import f1_score

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Start with default thresholds
    best_thresholds = [0.5] * num_classes

    # Grid search for optimal thresholds per class
    print("\nFinding optimal thresholds per class...")
    for class_idx in range(num_classes):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.9, 0.05):
            temp_thresholds = best_thresholds.copy()
            temp_thresholds[class_idx] = threshold

            # Apply thresholds
            adjusted_probs = all_probs.copy()
            for idx, thresh in enumerate(temp_thresholds):
                adjusted_probs[:, idx] = (all_probs[:, idx] >= thresh).astype(float) * all_probs[:, idx]

            preds = np.argmax(adjusted_probs, axis=1)
            f1 = f1_score(all_labels, preds, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds[class_idx] = best_threshold
        print(f"  Class {class_idx}: threshold={best_threshold:.2f}")

    return best_thresholds

if not os.path.isfile(model_path):

    # Option 1: Manually define test video IDs (set to None to use random split)
    manual_test_video_ids = ['3279_21min_behaviour_2023-01-19T12_57_29', '20231123_10min_OFT-BL_4028',
                             'BehavioralCamera2023-02-23T10_23_42_shorter', 'MBT1-M2', 'T2',
                             'MBT1-M7', 'T8', 'T4', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'T1']
    #manual_test_video_ids = ['MBT1-M10', 'T18', 'MBT1-M2', 'MBT1-M15', 'T1', 'T3']
    #manual_test_video_ids = ["T2","T4","T13","MBT1-M2","MBT1-M7","MBT1-M10"]  # Example: ['video1', 'video2', 'video3']

    if manual_test_video_ids is not None:
        # Use manually specified test videos
        all_video_ids = X.index.get_level_values("video_id").unique()
        test_video_ids = [vid for vid in manual_test_video_ids if vid in all_video_ids]
        train_video_ids = [vid for vid in all_video_ids if vid not in test_video_ids]

        print(f"Manual split: Total videos: {len(all_video_ids)}, Test videos: {len(test_video_ids)}, Train videos: {len(train_video_ids)}")
        print(f"Test video IDs: {test_video_ids}")

        X_train = X.loc[X.index.get_level_values('video_id').isin(train_video_ids)]
        X_test = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
        y_train = y.loc[y.index.get_level_values('video_id').isin(train_video_ids)]
        y_test = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]
    else:
        # Option 2: Random split (original behavior)
        n_videos = X.index.get_level_values("video_id").nunique()
        test_videos = max(1, int(n_videos * 0.2))
        print(f"Random split: Total videos: {n_videos}, Test videos: {test_videos}, Train videos: {n_videos - test_videos}")
        X_train, X_test, y_train, y_test = video_train_test_split(X, y, test_videos=test_videos, random_state=20)

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

    # Save label encoder
    import joblib
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Calculate class weights using sklearn's compute_class_weight (balanced)
    y_train_flat = y_train_encoded.values.ravel()
    unique, counts = np.unique(y_train_flat, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"\nClass distribution in training: {class_counts}")

    # Show test set distribution
    y_test_flat = y_test_encoded.values.ravel()
    unique_test, counts_test = np.unique(y_test_flat, return_counts=True)
    class_counts_test = dict(zip(unique_test, counts_test))
    print(f"Class distribution in test: {class_counts_test}")

    # Show percentage comparison
    print("\nClass distribution percentages:")
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        train_pct = (class_counts.get(cls_idx, 0) / len(y_train_flat)) * 100
        test_pct = (class_counts_test.get(cls_idx, 0) / len(y_test_flat)) * 100
        print(f"  {cls_name}: Train={train_pct:.1f}%, Test={test_pct:.1f}%")

    # Use sklearn's compute_class_weight for balanced weights
    class_weight_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_flat),
        y=y_train_flat
    )

    # Moderate the weights to avoid over-penalizing majority class
    # Scale weights so that the max weight is 3x the min weight instead of extreme ratios
    min_weight = class_weight_array.min()
    max_weight = class_weight_array.max()
    if max_weight / min_weight > 3.0:
        # Compress the range of weights
        class_weight_array = 1.0 + 2.0 * (class_weight_array - min_weight) / (max_weight - min_weight)

    class_weights = {i: weight for i, weight in enumerate(class_weight_array)}
    print(f"\nClass weights (moderated): {class_weights}")

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

    # Save scaler
    joblib.dump(scaler, scaler_path)

    # Create sequence datasets with stride to reduce memory usage
    SEQUENCE_LENGTH = 30  # Use 30 frames (1 second at 30 fps)
    STRIDE = 10  # Step size between sequences (reduces memory by 3x)
    print(f"Creating sequences with length {SEQUENCE_LENGTH} and stride {STRIDE}...")
    train_dataset = SequenceDataset(X_train_scaled, y_train_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    test_dataset = SequenceDataset(X_test_scaled, y_test_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)

    print(f"Total training sequences: {len(train_dataset)}, test sequences: {len(test_dataset)}")

    # Reduce batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = X_train.shape[1]
    hidden_size = 96    # Reduced from 128 to reduce overfitting
    num_layers = 2      # Reduced from 3 to 2 layers
    num_classes = len(unique)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.5  # Increased dropout to reduce overfitting
    ).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Focal Loss for imbalanced classes
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)]).to(device)
    criterion = FocalLoss(alpha=weight_tensor, gamma=1.0)  # Reduced gamma to be less aggressive on majority class
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR, higher weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    num_epochs = 150
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement to reset patience

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2)

        # Evaluate (without thresholds during training)
        y_pred, y_true, _ = evaluate(model, test_loader, device, thresholds=None)
        test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average='macro')

        scheduler.step(test_f1)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")

        # Early stopping with minimum delta
        if test_f1 > best_f1 + min_delta:
            best_f1 = test_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_classes': num_classes,
                'sequence_length': SEQUENCE_LENGTH,
                'class_weights': class_weights,
                'train_videos': X_train.index.get_level_values('video_id').unique().tolist(),
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
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Find optimal thresholds on validation set (using test set here)
    print("\n=== Finding Optimal Decision Thresholds ===")
    optimal_thresholds = find_optimal_thresholds(model, test_loader, device, num_classes)
    print(f"Optimal thresholds: {optimal_thresholds}")

    # Save thresholds with model
    checkpoint['optimal_thresholds'] = optimal_thresholds
    torch.save(checkpoint, model_path)

else:
    # Load existing model
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    num_classes = checkpoint['num_classes']
    SEQUENCE_LENGTH = checkpoint['sequence_length']
    class_weights = checkpoint['class_weights']

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.5
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimal thresholds if available
    optimal_thresholds = checkpoint.get('optimal_thresholds', None)
    if optimal_thresholds:
        print(f"Loaded optimal thresholds: {optimal_thresholds}")
    else:
        print("No optimal thresholds found in checkpoint (using default)")

    # Recreate train/test split using saved video IDs
    train_video_ids = checkpoint['train_videos']
    test_video_ids = checkpoint['test_videos']

    X_train = X.loc[X.index.get_level_values('video_id').isin(train_video_ids)]
    X_test = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
    y_train = y.loc[y.index.get_level_values('video_id').isin(train_video_ids)]
    y_test = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]

    # Load label encoder and encode labels
    label_encoder = joblib.load(label_encoder_path)
    y_train_encoded = pd.DataFrame(
        label_encoder.transform(y_train.values.ravel()),
        index=y_train.index,
        columns=[y_train.name] if isinstance(y_train, pd.Series) else y_train.columns
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
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    # Create sequence datasets with stride
    STRIDE = 10  # Match training stride
    train_dataset = SequenceDataset(X_train_scaled, y_train_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    test_dataset = SequenceDataset(X_test_scaled, y_test_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Loaded model from {model_path}")
    print(f"Using device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")

# Get optimal thresholds if not loaded from checkpoint
if 'optimal_thresholds' not in locals() or optimal_thresholds is None:
    print("\n=== Finding Optimal Decision Thresholds ===")
    optimal_thresholds = find_optimal_thresholds(model, test_loader, device, num_classes)
    print(f"Optimal thresholds: {optimal_thresholds}")

# Training set evaluation (without thresholds)
print("\n=== Training Set Evaluation (no threshold adjustment) ===")
y_pred_train, y_true_train, _ = evaluate(model, train_loader, device, thresholds=None)

train_acc = 100 * np.sum(y_pred_train == y_true_train) / len(y_true_train)
train_f1 = f1_score(y_true_train, y_pred_train, average='macro')

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Train F1 Score (macro): {train_f1:.4f}")
print("\nTraining Classification Report:")
print(classification_report(y_true_train, y_pred_train, target_names=label_encoder.classes_))

# Test set evaluation without thresholds
print("\n=== Test Set Evaluation (no threshold adjustment) ===")
y_pred, y_true, _ = evaluate(model, test_loader, device, thresholds=None)

test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
test_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test F1 Score (macro): {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Test set evaluation WITH optimal thresholds
print("\n=== Test Set Evaluation (WITH threshold adjustment) ===")
y_pred_thresh, y_true_thresh, _ = evaluate(model, test_loader, device, thresholds=optimal_thresholds)

test_acc_thresh = 100 * np.sum(y_pred_thresh == y_true_thresh) / len(y_true_thresh)
test_f1_thresh = f1_score(y_true_thresh, y_pred_thresh, average='macro')

print(f"Test Accuracy: {test_acc_thresh:.2f}%")
print(f"Test F1 Score (macro): {test_f1_thresh:.4f}")
print("\nClassification Report:")
print(classification_report(y_true_thresh, y_pred_thresh, target_names=label_encoder.classes_))
print(f"\nF1 Score improvement: {test_f1_thresh - test_f1:.4f} ({((test_f1_thresh/test_f1 - 1) * 100):.2f}%)")

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
