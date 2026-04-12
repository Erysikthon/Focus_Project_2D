"""
CNN-Transformer Pipeline

Architecture:
1. CNN Feature Extractor: Extracts spatial features from individual video frames
2. Transformer: Captures temporal dependencies across frame sequences
3. Classification Head: Predicts behavior labels (per-frame)

Input: Raw video frames from rotated_videos folder
Output: Behavior classification per frame(sequence??maybe better)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================================
# CNN Feature Extractor (mostly copied from TCNN.py)
# ============================================================================

class ResBlock2D(nn.Module):
    """2D Residual Block for spatial feature extraction"""
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv_block(x))


class CNNFeatureExtractor(nn.Module):
    """
    CNN to extract spatial features from individual frames
    Input: (batch, 1, H, W) - grayscale frames
    Output: (batch, feature_dim) - feature vector per frame
    """
    def __init__(self, feature_dim=512):
        super().__init__()

        # Initial convolution
        # Input (H=142, W=76) -> after two stride-2 convs + maxpool -> (H=18, W=9)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ResBlock population 1
        self.res_blocks_1 = nn.Sequential(*[ResBlock2D(64) for _ in range(3)])

        # Transition layer -> (H=9, W=5)
        self.transition_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ResBlock population 2
        self.res_blocks_2 = nn.Sequential(*[ResBlock2D(64) for _ in range(3)])

        # Transition layer -> (H=5, W=3)
        self.transition_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Feature projection (128 * 5 * 3 = 1920 spatial positions preserved)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        x: (batch, 1, H, W)
        returns: (batch, feature_dim)
        """
        x = self.initial_conv(x)
        x = self.res_blocks_1(x)
        x = self.transition_1(x)
        x = self.res_blocks_2(x)
        x = self.transition_2(x)
        x = self.fc(x)
        return x


# ============================================================================
# Positional Encoding for Transformer
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Combined CNN-Transformer Model
# ============================================================================

class CNNTransformerClassifier(nn.Module):
    """
    Combined CNN-Transformer for per-frame video behavior classification

    Architecture:
    1. CNN extracts features from each frame independently
    2. Transformer processes temporal sequence of CNN features
    3. Classification head predicts behavior for each frame (stride 10 -> 1 frame in 3 diff sequences, majority voting)
    """
    def __init__(self, cnn_feature_dim=512, d_model=512, nhead=8,
                 num_layers=4, num_classes=5, dim_feedforward=2048, dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(feature_dim=cnn_feature_dim)

        # Project CNN features to transformer dimension if needed
        self.feature_projection = nn.Linear(cnn_feature_dim, d_model) if cnn_feature_dim != d_model else nn.Identity()

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

        # Per-frame classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 1, H, W) - sequence of grayscale frames
        returns: (batch, seq_len, num_classes) - per-frame classification logits
        """
        batch_size, seq_len, c, h, w = x.shape

        # Extract CNN features for each frame
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, c, h, w)  # (batch*seq_len, 1, H, W)
        cnn_features = self.cnn(x)  # (batch*seq_len, cnn_feature_dim)

        # Reshape back to sequence
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # (batch, seq_len, cnn_feature_dim)

        # Project to transformer dimension
        x = self.feature_projection(cnn_features)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Per-frame classification
        logits = self.classifier(x)  # (batch, seq_len, num_classes)

        return logits


# ============================================================================
# Video Dataset
# ============================================================================

class VideoSequenceDataset(Dataset):
    """
    Lazy-loading dataset for video sequences (loads frames on-demand to save memory): otherwise crashes :C
    Returns per-frame labels for behavior transition detection
    """
    def __init__(self, video_folder, label_folder, video_ids, sequence_length=30,
                 stride=10, img_size=(76, 142),
                 behavior_names=None):
        """
        Args:
            video_folder: Path to folder containing .mp4 files
            label_folder: Path to folder containing label CSV files
            video_ids: List of video IDs (filenames without extension)
            sequence_length: Number of frames per sequence
            stride: Step size between sequences
            img_size: (width, height) - original video dimensions
        """
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size

        # Store metadata about sequences (not the actual frames)
        self.sequence_info = []  # List of (video_id, start_frame)
        self.labels = []  # For compatibility (will store first frame label of each sequence)
        self.label_cache = {}  # Cache video labels to avoid reloading CSV files
        self.behavior_names = behavior_names  # Global behavior list; None = infer per-video (legacy)

        print(f"Indexing sequences from {len(video_ids)} videos...")
        self._index_sequences(video_ids)

    def _index_sequences(self, video_ids):
        """Create index of sequences without loading video frames"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
            label_path = os.path.join(self.label_folder, f"{video_id}.csv")

            # Check if files exist
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Labels not found: {label_path}")
                continue

            # Load labels
            try:
                labels_df = pd.read_csv(label_path)
                if self.behavior_names is not None:
                    # Align to global behavior list: missing columns become all-zero (background)
                    for col in self.behavior_names:
                        if col not in labels_df.columns:
                            labels_df[col] = 0
                    video_labels = labels_df[self.behavior_names].values.argmax(axis=1)
                else:
                    behavior_columns = [col for col in labels_df.columns if col not in ['Unnamed: 0', 'frame']]
                    if len(behavior_columns) == 0:
                        print(f"Warning: Invalid label format for {video_id}")
                        continue
                    video_labels = labels_df[behavior_columns].values.argmax(axis=1)
                self.label_cache[video_id] = video_labels  # Cache labels for this video
            except Exception as e:
                print(f"Error loading labels for {video_id}: {e}")
                continue

            # Get video frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Index sequences (don't load frames yet)
            for start_idx in range(0, min(total_frames, len(video_labels)) - self.sequence_length + 1, self.stride):
                first_frame_label = video_labels[start_idx]
                self.sequence_info.append((video_id, start_idx))
                self.labels.append(first_frame_label)

        print(f"Indexed {len(self.sequence_info)} sequences (per-frame labels)")

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        """Load video frames and per-frame labels on-demand"""
        video_id, start_frame = self.sequence_info[idx]

        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        # Use cached labels instead of reloading CSV
        video_labels = self.label_cache[video_id]

        # Get labels for this sequence (all frames in the sequence)
        sequence_labels = video_labels[start_frame:start_frame + self.sequence_length]

        # Open video and load the specific sequence
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # If we can't read a frame, use a black frame
                frames.append(np.zeros(self.img_size[::-1], dtype=np.uint8))  # img_size is (W,H), array needs (H,W)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)  # Use original size, no resize

        cap.release()

        # Convert to numpy array
        frames = np.array(frames, dtype=np.float32)

        # Normalize to [-0.5, 0.5] (inverted, matching VideoDataSet/TCNN convention)
        frames = -(frames / 255.0 - 0.5)

        # Add channel dimension: (seq_len, H, W) -> (seq_len, 1, H, W)
        frames = frames[:, np.newaxis, :, :]

        return torch.FloatTensor(frames), torch.LongTensor(sequence_labels)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch with per-frame predictions"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in tqdm(dataloader, desc="Training"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)  # (batch, seq_len)

        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, seq_len, num_classes)

        # Reshape for loss computation
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(batch_size * seq_len, num_classes)
        labels_flat = batch_y.view(batch_size * seq_len)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_consensus=True, return_per_video=False):
    """
    Evaluate model with per-frame predictions

    Arguments:
        use_consensus: If True, uses majority voting across overlapping sequences for each unique frame
                      If False, treats all predictions independently (inflates metrics: 3 times the same frame predicted kinda)
    """
    model.eval()

    if not use_consensus:
        # Original behavior: treat all predictions independently
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader, desc="Evaluating"):
                batch_X = batch_X.to(device)
                outputs = model(batch_X)  # (batch, seq_len, num_classes)
                _, predicted = torch.max(outputs, 2)  # (batch, seq_len)
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.numpy().flatten())
        return np.array(all_preds), np.array(all_labels)

    # Consensus voting: aggregate predictions per unique (video_id, frame_idx)
    from collections import defaultdict

    frame_predictions = defaultdict(list)  # {(video_id, frame_idx): [probs1, probs2, ...]}
    frame_labels = {}  # {(video_id, frame_idx): true_label}

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)  # (batch, seq_len, num_classes)
            probs = torch.softmax(outputs, dim=2)  # (batch, seq_len, num_classes)

            probs_np = probs.cpu().numpy()
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

                    frame_predictions[key].append(probs_np[b, frame_offset])  # shape: (num_classes,)
                    frame_labels[key] = labels_np[b, frame_offset]  # Same label from all sequences

    # Apply majority voting
    consensus_preds = []
    consensus_labels = []
    per_video_data = defaultdict(lambda: {'preds': [], 'labels': []})

    for key in sorted(frame_predictions.keys()):
        video_id, frame_idx = key
        preds = frame_predictions[key]
        consensus_pred = np.argmax(np.sum(preds, axis=0))

        consensus_preds.append(consensus_pred)
        consensus_labels.append(frame_labels[key])

        per_video_data[video_id]['preds'].append(consensus_pred)
        per_video_data[video_id]['labels'].append(frame_labels[key])

    # Apply minimum duration filter then gap-filling per video
    for video_id in per_video_data:
        filtered = apply_min_duration_filter(per_video_data[video_id]['preds'])
        filtered = apply_gap_fill(filtered)
        per_video_data[video_id]['preds'] = filtered.tolist()

    # Reconstruct flat arrays in sorted key order from filtered per-video data
    video_frame_counters = defaultdict(int)
    consensus_preds = []
    for key in sorted(frame_predictions.keys()):
        video_id, _ = key
        i = video_frame_counters[video_id]
        consensus_preds.append(per_video_data[video_id]['preds'][i])
        video_frame_counters[video_id] += 1

    if return_per_video:
        return np.array(consensus_preds), np.array(consensus_labels), dict(per_video_data)
    return np.array(consensus_preds), np.array(consensus_labels)

# Final Label Smoothing
def apply_min_duration_filter(preds, min_duration=5, background_class=0):
    """
    Remove short predicted runs of non-background classes.
    Any contiguous run shorter than min_duration frames is replaced by
    the preceding class (or background if at the start).
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        cls = preds[i]
        if cls == background_class:
            i += 1
            continue
        # Find end of this run
        j = i
        while j < len(preds) and preds[j] == cls:
            j += 1
        run_length = j - i
        if run_length < min_duration:
            replacement = preds[i - 1] if i > 0 else background_class
            preds[i:j] = replacement
        i = j
    return preds


def apply_gap_fill(preds, max_gap=5, background_class=0):
    """
    Fill short background gaps between identical behaviors.
    If background appears for <= max_gap frames between the same behavior on both sides,
    replace the background with that behavior.
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        if preds[i] != background_class:
            i += 1
            continue
        # Found a background run — find its extent
        j = i
        while j < len(preds) and preds[j] == background_class:
            j += 1
        gap_length = j - i
        # Check if gap is short enough and flanked by the same behavior
        if gap_length <= max_gap and i > 0 and j < len(preds):
            before = preds[i - 1]
            after = preds[j]
            if before == after and before != background_class:
                preds[i:j] = before
        i = j
    return preds

# Behavior Counting
def count_behavior_instances(predictions, behavior_label):
    """Count behavior instances by detecting transitions to the behavior (0->1 transitions in binary mask)."""
    behavior_mask = (predictions == behavior_label).astype(int)
    changes = np.diff(behavior_mask, prepend=0)
    return int((changes > 0.5).sum())


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Configuration
    DATASET_VERSION = "org_v15_val_set"
    VIDEO_FOLDER = "./data/rotated_videos_with_OFT"  #_with_OFT"
    LABEL_FOLDER = "./data/labels_with_OFT"  #_with_OFT"
    MODEL_PATH = f"./output_cnn_transformer/CNN_Transformer_{DATASET_VERSION}.pth"
    LABEL_ENCODER_PATH = f"./output_cnn_transformer/label_encoder_{DATASET_VERSION}.pkl"

    # Hyperparameters
    SEQUENCE_LENGTH = 30 #tried 60, worse. Avg behavior duration is 30
    TRAIN_STRIDE = 10       # stride for training dataset
    EPOCH_EVAL_STRIDE = 10  # stride for per-epoch evaluation during training (fast)
    FINAL_EVAL_STRIDE = 5   # stride for final evaluation (denser, ~6 votes/frame via consensus)
    IMG_SIZE = (76, 142)  # Original video dimensions (width, height)
    BATCH_SIZE = 32  # Increased from 16 for faster training
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001

    CNN_FEATURE_DIM = 512 #tried half, worse
    D_MODEL = 512 #tried half, worse
    NHEAD = 8
    NUM_LAYERS = 4 #tried 3, worse
    DIM_FEEDFORWARD = 2048 #tried half, worse
    DROPOUT = 0.3  # tried 0.5 (v8) and 0.4 (v9) — both worse on test

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get video IDs
    all_video_ids = [f.replace('.mp4', '') for f in os.listdir(VIDEO_FOLDER)
                     if f.endswith('.mp4')]

    # Balanced train / val / test split (from balance_train_val_test_split.py)
    # Val is used for early stopping; test is evaluated once on the final best model.
    train_video_ids = ['20231123_10min_OFT-BL_3919', '20231123_10min_OFT-BL_3961',
                       '20231123_10min_OFT-BL_3962', '20231123_10min_OFT-BL_3963',
                       '20231123_10min_OFT-BL_3964', '20231123_10min_OFT-BL_4029',
                       '3279_21min_behaviour_2023-01-19T12_57_29',
                       'BehavioralCamera2023-02-14T13_05_19_shorter',
                       'BehavioralCamera2023-02-14T15_22_37_shorter',
                       'BehavioralCamera2023-02-15T14_40_46_shorter',
                       'BehavioralCamera2023-02-23T10_23_42_shorter',
                       'BehavioralCamera2023-02-23T15_42_37_shorter',
                       'BehavioralCamera2023-02-24T11_06_53_shorter',
                       'BehavioralCamera2023-03-09T10_37_32', 'BehavioralCamera2023-03-09T12_08_14',
                       'BehavioralCamera2023-03-09T12_34_50', 'BehavioralCamera2023-03-09T13_02_04',
                       'BehavioralCamera2023-03-09T14_30_45',
                       'MBT1-M11', 'MBT1-M14', 'MBT1-M15', 'MBT1-M18', 'MBT1-M2', 'MBT1-M6',
                       'OFT_left_1', 'OFT_left_11', 'OFT_left_12', 'OFT_left_13', 'OFT_left_15',
                       'OFT_left_17', 'OFT_left_19', 'OFT_left_4', 'OFT_left_6', 'OFT_left_7',
                       'OFT_left_8', 'OFT_left_9',
                       'T1', 'T10', 'T12', 'T14', 'T16', 'T17', 'T18', 'T19',
                       'T3', 'T4', 'T5', 'T6', 'T8', 'T9']
    val_video_ids   = ['20231123_10min_OFT-BL_4025', '20231123_10min_OFT-BL_4028',
                       'BehavioralCamera2023-02-18T10_33_06_shorter',
                       'BehavioralCamera2023-02-19T14_53_53_shorter',
                       'BehavioralCamera2023-03-09T11_41_07',
                       'MBT1-M10', 'MBT1-M3',
                       'OFT_left_16', 'OFT_left_18', 'OFT_left_2',
                       'T11', 'T15']
    test_video_ids  = ['3278_21min_behaviour_2023-01-19T11_08_30',
                       'BehavioralCamera2023-02-18T12_37_43_shorter',
                       'BehavioralCamera2023-03-09T11_04_40',
                       'MBT1-M7',
                       'OFT_left_10', 'OFT_left_14', 'OFT_left_20', 'OFT_left_21', 'OFT_left_3',
                       'T13', 'T2', 'T7']

    # Filter to videos actually present on disk
    train_video_ids = [v for v in train_video_ids if v in all_video_ids]
    val_video_ids   = [v for v in val_video_ids   if v in all_video_ids]
    test_video_ids  = [v for v in test_video_ids  if v in all_video_ids]

    print(f"Split: {len(train_video_ids)} train / {len(val_video_ids)} val / {len(test_video_ids)} test videos")
    print(f"Val IDs:  {val_video_ids}")
    print(f"Test IDs: {test_video_ids}")

    # Get behavior class names as union across all label files (handles videos with missing behaviors)
    all_behavior_cols = set()
    for vid in all_video_ids:
        lp = os.path.join(LABEL_FOLDER, f"{vid}.csv")
        if os.path.exists(lp):
            df = pd.read_csv(lp, nrows=0)  # only read header
            cols = [c for c in df.columns if c not in ['Unnamed: 0', 'frame']]
            all_behavior_cols.update(cols)
    behavior_names = sorted(all_behavior_cols)  # sorted for consistent ordering

    num_classes = len(behavior_names)
    print(f"Classes: {behavior_names}")
    print(f"Number of classes: {num_classes}")

    # Create datasets
    print(f"\nCreating training dataset from {len(train_video_ids)} videos...")
    train_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, train_video_ids,
        SEQUENCE_LENGTH, TRAIN_STRIDE, IMG_SIZE,
        behavior_names=behavior_names
    )
    print(f"Training dataset created: {len(train_dataset)} sequences")

    print(f"\nCreating epoch val dataset from {len(val_video_ids)} val videos (stride={EPOCH_EVAL_STRIDE})...")
    epoch_val_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, val_video_ids,
        SEQUENCE_LENGTH, EPOCH_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names
    )
    print(f"Epoch val dataset: {len(epoch_val_dataset)} sequences")

    print(f"\nCreating final eval datasets (stride={FINAL_EVAL_STRIDE})...")
    train_eval_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, train_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names
    )
    val_eval_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, val_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names
    )
    test_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, test_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names
    )
    print(f"Train eval dataset: {len(train_eval_dataset)} sequences")
    print(f"Val eval dataset:   {len(val_eval_dataset)} sequences")
    print(f"Test dataset:       {len(test_dataset)} sequences")

    # Convert labels to int (they're already integer indices from argmax)
    train_dataset.labels = [int(label) for label in train_dataset.labels]
    val_eval_dataset.labels = [int(label) for label in val_eval_dataset.labels]
    test_dataset.labels = [int(label) for label in test_dataset.labels]

    # Save behavior names for later use
    joblib.dump(behavior_names, LABEL_ENCODER_PATH)

    # Check if model already exists
    if os.path.isfile(MODEL_PATH):
        print(f"\n{'='*60}")
        print(f"Found existing model at: {MODEL_PATH}")
        print(f"Loading model instead of training...")
        print(f"{'='*60}\n")

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        # Initialize model with saved parameters
        model = CNNTransformerClassifier(
            cnn_feature_dim=checkpoint['cnn_feature_dim'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=DROPOUT
        ).to(device)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✓ Model loaded successfully!")

        # Print model architecture details
        print(f"\n{'='*60}")
        print(f"MODEL ARCHITECTURE")
        print(f"{'='*60}")
        print(f"CNN Feature Dimension:    {checkpoint['cnn_feature_dim']}")
        print(f"Transformer d_model:      {checkpoint['d_model']}")
        print(f"Number of attention heads: {checkpoint['nhead']}")
        print(f"Number of layers:         {checkpoint['num_layers']}")
        print(f"Feedforward dimension:    {checkpoint['dim_feedforward']}")
        print(f"Number of classes:        {checkpoint['num_classes']}")
        print(f"Dropout:                  {DROPOUT}")
        print(f"Total parameters:         {sum(p.numel() for p in model.parameters()):,}")

        # Print data/training hyperparameters
        print(f"\n{'='*60}")
        print(f"DATA & TRAINING HYPERPARAMETERS")
        print(f"{'='*60}")
        print(f"Sequence length:          {checkpoint['sequence_length']} frames ({checkpoint['sequence_length']/30:.1f} seconds)")
        print(f"Image size:               {checkpoint['img_size'][0]}×{checkpoint['img_size'][1]}")
        print(f"Batch size (current):     {BATCH_SIZE}")
        print(f"Device:                   {device}")

        # Print dataset info
        print(f"\n{'='*60}")
        print(f"DATASET INFO")
        print(f"{'='*60}")
        print(f"Training sequences:       {len(train_dataset)}")
        print(f"Val sequences:            {len(val_eval_dataset)}")
        print(f"Test sequences:           {len(test_dataset)}")
        print(f"Classes:                  {behavior_names}")
        print(f"{'='*60}\n")

        # Skip training, go directly to evaluation
        SKIP_TRAINING = True

    else:
        print(f"\nNo existing model found at: {MODEL_PATH}")
        print(f"Training new model...\n")

        # Create dataloaders (num_workers=4 for parallel data loading)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

        # Initialize model
        model = CNNTransformerClassifier(
            cnn_feature_dim=CNN_FEATURE_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT
        ).to(device)

        print(f"\nModel architecture:\n{model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        SKIP_TRAINING = False

    # Create epoch val dataloader (fast stride, used for early stopping during training)
    epoch_val_loader = DataLoader(epoch_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

    # Create final eval dataloaders (denser stride)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=2, pin_memory=True)
    val_eval_loader = DataLoader(val_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Class weights for imbalanced data (needed for training or info)
    unique, counts = np.unique(train_dataset.labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_samples = len(train_dataset.labels)
    n_classes = len(unique)

    # Print class distribution
    print(f"\n=== Class Distribution ===")
    for cls_idx, cls_name in enumerate(behavior_names):
        count = class_counts.get(cls_idx, 0)
        percentage = 100 * count / total_samples
        print(f"{cls_name}: {count} ({percentage:.2f}%)")

    # Calculate class weights (power of 0.7)
    class_weights = {}
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 0)
        if count > 0:
            class_weights[cls_idx] = (total_samples / (num_classes * count)) ** 0.7
        else:
            class_weights[cls_idx] = 1.0

    class_weights_array = np.array([class_weights[i] for i in range(num_classes)])

    print(f"\n=== Class Weights (power=0.7) ===")
    for cls_idx, cls_name in enumerate(behavior_names):
        weight = class_weights_array[cls_idx]
        count = class_counts.get(cls_idx, 0)
        print(f"{cls_name}: {weight:.3f} (count: {count})")

    weight_tensor = torch.FloatTensor(class_weights_array).to(device)

    # Training loop (skip if model already exists)
    if not SKIP_TRAINING:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # tried 0.05 — no improvement

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_f1 = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)

            # Evaluate val set with consensus voting (fast stride for per-epoch speed)
            y_pred, y_true = evaluate(model, epoch_val_loader, device, use_consensus=True)
            val_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
            val_f1 = f1_score(y_true, y_pred, average='macro')

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc (consensus): {val_acc:.2f}%, Val F1 (consensus): {val_f1:.4f}")

            # Save best model based on val F1 (test set not touched during training)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'cnn_feature_dim': CNN_FEATURE_DIM,
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'dim_feedforward': DIM_FEEDFORWARD,
                    'num_classes': num_classes,
                    'sequence_length': SEQUENCE_LENGTH,
                    'img_size': IMG_SIZE
                }, MODEL_PATH)
                print(f"→ New best model saved! Val F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Load best model from training
        print("\nLoading best model from training...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    log("\n" + "="*60)
    log("FINAL TRAINING SET EVALUATION")
    log("="*60)

    y_pred_train, y_true_train = evaluate(model, train_eval_loader, device)
    log("\nClassification Report:")
    log(classification_report(y_true_train, y_pred_train, target_names=behavior_names))

    # Training confusion matrix
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_train_pct = cm_train.astype(float) / cm_train.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=behavior_names,
                yticklabels=behavior_names)
    plt.title(f'Training Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./output_cnn_transformer/conf_matrix_train_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')

    log("\n" + "="*60)
    log("FINAL VALIDATION SET EVALUATION")
    log("="*60)

    y_pred_val, y_true_val = evaluate(model, val_eval_loader, device)
    log("\nClassification Report:")
    log(classification_report(y_true_val, y_pred_val, target_names=behavior_names))

    cm_val = confusion_matrix(y_true_val, y_pred_val)
    cm_val_pct = cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_val_pct, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=behavior_names,
                yticklabels=behavior_names)
    plt.title(f'Validation Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./output_cnn_transformer/conf_matrix_val_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')

    log("\n" + "="*60)
    log("FINAL TEST SET EVALUATION")
    log("="*60)

    y_pred, y_true, per_video_data = evaluate(model, test_loader, device, return_per_video=True)
    log("\nClassification Report:")
    log(classification_report(y_true, y_pred, target_names=behavior_names))

    log("\n" + "="*60)
    log("PER-VIDEO TEST EVALUATION")
    log("="*60)
    for video_id in sorted(per_video_data.keys()):
        v_preds = np.array(per_video_data[video_id]['preds'])
        v_labels = np.array(per_video_data[video_id]['labels'])
        v_f1 = f1_score(v_labels, v_preds, average='macro', zero_division=0)
        log(f"\n--- {video_id} (macro F1: {v_f1:.3f}) ---")
        log(classification_report(v_labels, v_preds, target_names=behavior_names, labels=range(len(behavior_names)), zero_division=0))

    # ============================================================================
    # BEHAVIOUR INSTANCE COUNTS
    # ============================================================================

    log("\n" + "="*60)
    log("BEHAVIOUR INSTANCE COUNTS - TRAIN SET")
    log("="*60)
    log(f"{'Behavior':<25} {'True Count':<15} {'Predicted Count':<15}")
    log("="*55)
    for cls_idx, cls_name in enumerate(behavior_names):
        true_count = count_behavior_instances(y_true_train, cls_idx)
        pred_count = count_behavior_instances(y_pred_train, cls_idx)
        log(f"{cls_name:<25} {true_count:<15} {pred_count:<15}")

    log("\n" + "="*60)
    log("BEHAVIOUR INSTANCE COUNTS - TEST SET")
    log("="*60)
    log(f"{'Behavior':<25} {'True Count':<15} {'Predicted Count':<15}")
    log("="*55)
    true_counts = {}
    pred_counts = {}
    for cls_idx, cls_name in enumerate(behavior_names):
        true_count = count_behavior_instances(y_true, cls_idx)
        pred_count = count_behavior_instances(y_pred, cls_idx)
        true_counts[cls_name] = true_count
        pred_counts[cls_name] = pred_count
        log(f"{cls_name:<25} {true_count:<15} {pred_count:<15}")

    # Bar chart for non-background behaviors
    plot_behavior_indices = list(range(1, num_classes))  # skip background (index 0)
    plot_behavior_names = [behavior_names[i] for i in plot_behavior_indices]

    if plot_behavior_names:
        fig, axes = plt.subplots(1, len(plot_behavior_names), figsize=(5 * len(plot_behavior_names), 5))
        if len(plot_behavior_names) == 1:
            axes = [axes]

        colors_true = '#59a89c'
        colors_pred = '#a559aa'

        for idx, cls_name in enumerate(plot_behavior_names):
            true_val = true_counts[cls_name]
            pred_val = pred_counts[cls_name]
            x_pos = np.array([0, 1])
            values = [true_val, pred_val]
            bars = axes[idx].bar(x_pos, values, color=[colors_true, colors_pred], width=0.6)

            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2., height,
                               f'{int(height)}', ha='center', va='bottom', fontsize=13, fontweight='bold')

            axes[idx].set_title(f'{cls_name}', fontsize=15, fontweight='bold', pad=10)
            axes[idx].set_ylabel('Instance Count', fontsize=14, labelpad=10)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(['True', 'Predicted'], fontsize=14)
            axes[idx].tick_params(axis='y', labelsize=14)
            axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
            axes[idx].set_ylim([0, max(values) * 1.15 if max(values) > 0 else 1])

        fig.suptitle(f'Behaviour Instance Counts (True vs Predicted) - {DATASET_VERSION}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'./output_cnn_transformer/behaviour_instance_count_{DATASET_VERSION}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\nBehaviour instance count plot saved!")

    # Per-video scatterplot
    if per_video_data and plot_behavior_names:
        video_counts = {cls_name: {'actual': [], 'predicted': []} for cls_name in plot_behavior_names}

        for video_id in sorted(per_video_data.keys()):
            v_preds = np.array(per_video_data[video_id]['preds'])
            v_labels = np.array(per_video_data[video_id]['labels'])
            for cls_idx, cls_name in zip(plot_behavior_indices, plot_behavior_names):
                video_counts[cls_name]['actual'].append(count_behavior_instances(v_labels, cls_idx))
                video_counts[cls_name]['predicted'].append(count_behavior_instances(v_preds, cls_idx))

        fig, axes = plt.subplots(1, len(plot_behavior_names), figsize=(5 * len(plot_behavior_names), 5))
        if len(plot_behavior_names) == 1:
            axes = [axes]

        for idx, cls_name in enumerate(plot_behavior_names):
            actual = np.array(video_counts[cls_name]['actual'])
            predicted = np.array(video_counts[cls_name]['predicted'])

            axes[idx].scatter(actual, predicted, alpha=0.6, s=100)
            max_val = max(actual.max(), predicted.max()) if (actual.max() > 0 or predicted.max() > 0) else 1
            axes[idx].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect Agreement')
            axes[idx].set_xlabel('Actual Instance Count', fontsize=14)
            axes[idx].set_ylabel('Predicted Instance Count', fontsize=14)
            axes[idx].set_title(f'{cls_name}', fontsize=17, fontweight='bold', pad=10)
            axes[idx].legend(fontsize=10.5)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal', adjustable='box')

        fig.suptitle(f'Actual vs Predicted Instance Counts Per Video - {DATASET_VERSION}',
                     fontsize=20, fontweight='bold', y=1)
        plt.tight_layout()
        plt.savefig(f'./output_cnn_transformer/instance_count_per_video_{DATASET_VERSION}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\nInstance count per video scatterplot saved!")

    # Test confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=behavior_names,
                yticklabels=behavior_names)
    plt.title(f'Test Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./output_cnn_transformer/conf_matrix_test_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')

    results_path = f'./output_cnn_transformer/evaluation_{DATASET_VERSION}.txt'
    with open(results_path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"\nEvaluation results saved to {results_path}")

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
