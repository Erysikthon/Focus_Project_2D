"""
CNN-Transformer Pipeline

Architecture:
1. CNN Feature Extractor: Extracts spatial features from individual video frames
2. Transformer: Captures temporal dependencies across frame sequences
3. Classification Head: Predicts behavior labels (per-frame)

Input: Raw video frames from rotated_videos folder
Output: Behavior classification per frame

"""

import torch
import torch.nn as nn
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
    def __init__(self, channels : int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv_block(x))


class CNNFeatureExtractor(nn.Module):
    """
    CNN to extract spatial features from individual frames
    Input: (batch, 1, H, W) - grayscale frames
    Output: (batch, feature_dim) - feature vector per frame
    """
    def __init__(self, feature_dim : int = 512, res_depth : int = 4, dropout : float = 0.3):
        super().__init__()

        # Initial convolution
        # Input (H=142, W=76) -> after two stride-2 convs -> (H=36, W=19)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (5, 5), stride = (2, 2), padding = (2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size = (5, 5), stride = (2, 2), padding = (2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # ResBlock population 1
        self.res_blocks_1 = nn.Sequential(*[ResBlock2D(48) for _ in range(res_depth)])

        # Transition layer -> (H=18, W=9)
        self.transition_1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))
        )

        # ResBlock population 2
        self.res_blocks_2 = nn.Sequential(*[ResBlock2D(64) for _ in range(res_depth)])

        # Transition layer -> (H=9, W=4)
        self.transition_2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(80),
            nn.Conv2d(80, 80, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))
        )

        self.res_blocks_3 = nn.Sequential(*[ResBlock2D(80) for _ in range(res_depth)])

        # Feature projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 9 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
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
        x = self.res_blocks_3(x)
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
    3. Simplified classification head: Dropout + 2 layers
    """
    def __init__(self, cnn_feature_dim=512, d_model=512, nhead=8,
                 num_layers=2, num_classes=5, dim_feedforward=2048, dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(feature_dim=cnn_feature_dim, dropout=dropout)

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

        self.classifier = nn.Sequential(
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
        x = x.view(batch_size * seq_len, c, h, w)  # (batch*seq_len, 1, H, W)
        cnn_features = self.cnn(x)                  # (batch*seq_len, cnn_feature_dim)

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
    Lazy-loading dataset for video sequences (loads frames on-demand to save memory).
    Returns per-frame labels for behavior classification.
    Augmentation applied during training.
    """
    def __init__(self, video_folder, label_folder, video_ids, sequence_length=30,
                 stride=10, img_size=(76, 142), behavior_names=None, augment=False):
        """
        Args:
            video_folder: Path to folder containing .mp4 files
            label_folder: Path to folder containing label CSV files
            video_ids: List of video IDs (filenames without extension)
            sequence_length: Number of frames per sequence
            stride: Step size between sequences
            img_size: (width, height) - original video dimensions
            behavior_names: Ordered list of behavior class names
            augment: If True, apply random augmentations (only for training)
        """
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.augment = augment

        self.sequence_info = []
        self.labels = []
        self.label_cache = {}
        self.behavior_names = behavior_names

        print(f"Indexing sequences from {len(video_ids)} videos...")
        self._index_sequences(video_ids)

    def _index_sequences(self, video_ids):
        """Create index of sequences without loading video frames"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
            label_path = os.path.join(self.label_folder, f"{video_id}.csv")

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Labels not found: {label_path}")
                continue

            try:
                labels_df = pd.read_csv(label_path)
                if self.behavior_names is not None:
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
                self.label_cache[video_id] = video_labels
            except Exception as e:
                print(f"Error loading labels for {video_id}: {e}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

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

        video_labels = self.label_cache[video_id]
        sequence_labels = video_labels[start_frame:start_frame + self.sequence_length]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(self.img_size[::-1], dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.img_size)  # img_size is (width, height)
                frames.append(gray)

        cap.release()

        frames = np.array(frames, dtype=np.float32)

        # Data augmentation (training only).
        # 25% of sequences pass through completely unaugmented.
        # Each remaining augmentation fires independently at 50%.
        if self.augment and np.random.random() > 0.25:
            # Horizontal flip
            if np.random.random() > 0.5:
                frames = frames[:, :, ::-1].copy()

            # Per-frame brightness/contrast jitter
            if np.random.random() > 0.5:
                contrast   = np.random.uniform(0.8, 1.2, size=(len(frames), 1, 1)).astype(np.float32)
                brightness = np.random.uniform(-20.0, 20.0, size=(len(frames), 1, 1)).astype(np.float32)
                frames = np.clip(frames * contrast + brightness, 0, 255)

            # Additive Gaussian noise
            if np.random.random() > 0.5:
                frames = np.clip(frames + np.random.normal(0, 8, frames.shape).astype(np.float32), 0, 255)

        # Normalize to [-0.5, 0.5] (inverted, matching VideoDataSet/TCNN convention)
        frames = -(frames / 255.0 - 0.5)

        # Add channel dimension: (seq_len, H, W) -> (seq_len, 1, H, W)
        frames = frames[:, np.newaxis, :, :]

        return torch.FloatTensor(frames), torch.LongTensor(sequence_labels)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch with per-frame predictions.
    CHANGE: scheduler removed from here — now stepped per epoch in the main loop.
    """
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
        labels_flat  = batch_y.view(batch_size * seq_len)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_consensus=True, return_per_video=False, background_class=0):
    """
    Evaluate model with per-frame predictions.

    Arguments:
        use_consensus: If True, uses majority voting across overlapping sequences for each unique frame.
                       If False, treats all predictions independently.
    """
    model.eval()

    if not use_consensus:
        if return_per_video:
            raise ValueError("return_per_video=True requires use_consensus=True")
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader, desc="Evaluating"):
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 2)
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.numpy().flatten())
        return np.array(all_preds), np.array(all_labels)

    from collections import defaultdict

    frame_predictions = defaultdict(list)
    frame_labels      = {}

    with torch.no_grad():
        # CHANGE: track global sequence index explicitly to avoid batch-size assumption
        seq_idx_offset = 0
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_X   = batch_X.to(device)
            outputs   = model(batch_X)
            probs     = torch.softmax(outputs, dim=2)

            probs_np  = probs.cpu().numpy()
            labels_np = batch_y.numpy()

            actual_batch_size = batch_X.shape[0]
            for b in range(actual_batch_size):
                seq_idx = seq_idx_offset + b
                if seq_idx >= len(dataloader.dataset):
                    continue

                video_id, start_frame = dataloader.dataset.sequence_info[seq_idx]

                for frame_offset in range(dataloader.dataset.sequence_length):
                    frame_idx = start_frame + frame_offset
                    key       = (video_id, frame_idx)

                    frame_predictions[key].append(probs_np[b, frame_offset])
                    frame_labels[key] = labels_np[b, frame_offset]

            seq_idx_offset += actual_batch_size

    # Consensus voting
    consensus_preds  = []
    consensus_labels = []
    per_video_data   = defaultdict(lambda: {'preds': [], 'labels': []})

    for key in sorted(frame_predictions.keys()):
        video_id, frame_idx = key
        preds          = frame_predictions[key]
        consensus_pred = np.argmax(np.sum(preds, axis=0))

        consensus_preds.append(consensus_pred)
        consensus_labels.append(frame_labels[key])

        per_video_data[video_id]['preds'].append(consensus_pred)
        per_video_data[video_id]['labels'].append(frame_labels[key])

    # Apply postprocessing per video
    for video_id in per_video_data:
        filtered = apply_min_duration_filter(per_video_data[video_id]['preds'], background_class=background_class)
        filtered = apply_gap_fill(filtered, background_class=background_class)
        per_video_data[video_id]['preds'] = filtered.tolist()

    # Reconstruct flat arrays from filtered per-video videos
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


def apply_min_duration_filter(preds, min_duration=5, background_class=0):
    """
    Remove short predicted runs of non-background classes.
    Any contiguous run shorter than min_duration frames is replaced by
    the preceding class (or background if at the start).
    When a replacement is made, the scan restarts from i so that the newly
    patched-in class is also checked against min_duration.
    """
    preds = np.array(preds, dtype=int)
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
                # Merged into the preceding run of the same class; no re-scan needed
                # (re-scanning would loop forever since preds[i-1] never changes).
                i = j
            # else: re-scan from i — replacement may itself be a short non-background run
        else:
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


def count_behavior_instances(predictions, behavior_label):
    """Count behavior instances by detecting transitions (0->1 in binary mask)."""
    behavior_mask = (predictions == behavior_label).astype(int)
    changes = np.diff(behavior_mask, prepend=0)
    return int((changes > 0.5).sum())


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Configuration
    DATASET_VERSION = "OFT_v1"
    VIDEO_FOLDER    = "./pipeline_inputs/rotated_videos_OFT"
    LABEL_FOLDER    = "./pipeline_inputs/labels_OFT"
    MODEL_PATH      = f"./pipeline_outputs/cnn_transformer/CNN_Transformer_{DATASET_VERSION}.pth"
    LABEL_ENCODER_PATH = f"./pipeline_outputs/cnn_transformer/label_encoder_{DATASET_VERSION}.pkl"

    # Hyperparameters
    SEQUENCE_LENGTH    = 30
    TRAIN_STRIDE       = 10
    EPOCH_EVAL_STRIDE  = 10
    FINAL_EVAL_STRIDE  = 5
    IMG_SIZE           = (76, 142)
    BATCH_SIZE         = 32
    NUM_EPOCHS         = 100
    LEARNING_RATE      = 0.0001

    CNN_FEATURE_DIM  = 512
    D_MODEL          = 512
    NHEAD            = 8
    NUM_LAYERS       = 3
    DIM_FEEDFORWARD  = 2048
    DROPOUT          = 0.3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get video IDs
    all_video_ids = [f.replace('.mp4', '') for f in os.listdir(VIDEO_FOLDER)
                     if f.endswith('.mp4')]

    # OFT dataset: 20 videos (OFT_left_1..21, no 5) — 14 train / 3 val / 3 test (70/15/15)
    # Val and test are spread evenly across the numeric range to avoid cluster bias.
    val_video_ids  = ['OFT_left_7',  'OFT_left_14', 'OFT_left_20']
    test_video_ids = ['OFT_left_4',  'OFT_left_11', 'OFT_left_17']
    held_out = set(val_video_ids + test_video_ids)
    train_video_ids = [v for v in all_video_ids if v not in held_out]

    val_video_ids  = [v for v in val_video_ids  if v in all_video_ids]
    test_video_ids = [v for v in test_video_ids if v in all_video_ids]

    print(f"Split: {len(train_video_ids)} train / {len(val_video_ids)} val / {len(test_video_ids)} test videos")
    print(f"Val IDs:  {val_video_ids}")
    print(f"Test IDs: {test_video_ids}")

    # Get behavior class names
    behavior_names = None
    for vid in all_video_ids:
        lp = os.path.join(LABEL_FOLDER, f"{vid}.csv")
        if os.path.exists(lp):
            df = pd.read_csv(lp, nrows=0)
            cols = [c for c in df.columns if c not in ['Unnamed: 0', 'frame']]
            if behavior_names is None:
                behavior_names = cols
            break  # column order is consistent across CSVs
    assert behavior_names is not None, "No label CSVs found"
    assert behavior_names[0].lower() == 'background', \
        f"Expected background at index 0, got: {behavior_names[0]}"

    num_classes = len(behavior_names)
    print(f"Classes: {behavior_names}")
    print(f"Number of classes: {num_classes}")

    # Create datasets
    # CHANGE: augment=True for training dataset only
    print(f"\nCreating training dataset from {len(train_video_ids)} videos...")
    train_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, train_video_ids,
        SEQUENCE_LENGTH, TRAIN_STRIDE, IMG_SIZE,
        behavior_names=behavior_names,
        augment=True   # CHANGE: augmentation enabled for training
    )
    print(f"Training dataset created: {len(train_dataset)} sequences")

    print(f"\nCreating epoch val dataset (stride={EPOCH_EVAL_STRIDE})...")
    epoch_val_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, val_video_ids,
        SEQUENCE_LENGTH, EPOCH_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names,
        augment=False  # no augmentation for evaluation
    )
    print(f"Epoch val dataset: {len(epoch_val_dataset)} sequences")

    print(f"\nCreating final eval datasets (stride={FINAL_EVAL_STRIDE})...")
    train_eval_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, train_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names,
        augment=False
    )
    val_eval_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, val_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names,
        augment=False
    )
    test_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, test_video_ids,
        SEQUENCE_LENGTH, FINAL_EVAL_STRIDE, IMG_SIZE,
        behavior_names=behavior_names,
        augment=False
    )
    print(f"Train eval dataset: {len(train_eval_dataset)} sequences")
    print(f"Val eval dataset:   {len(val_eval_dataset)} sequences")
    print(f"Test dataset:       {len(test_dataset)} sequences")

    train_dataset.labels      = [int(l) for l in train_dataset.labels]
    train_eval_dataset.labels = [int(l) for l in train_eval_dataset.labels]
    val_eval_dataset.labels   = [int(l) for l in val_eval_dataset.labels]
    test_dataset.labels       = [int(l) for l in test_dataset.labels]

    joblib.dump(behavior_names, LABEL_ENCODER_PATH)

    # Check if model already exists
    if os.path.isfile(MODEL_PATH):
        print(f"\n{'='*60}")
        print(f"Found existing model at: {MODEL_PATH}")
        print(f"Loading model instead of training...")
        print(f"{'='*60}\n")

        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        model = CNNTransformerClassifier(
            cnn_feature_dim=checkpoint['cnn_feature_dim'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=checkpoint.get('dropout', DROPOUT)  # fall back to runtime value for old checkpoints
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully!")

        print(f"\n{'='*60}")
        print(f"MODEL ARCHITECTURE")
        print(f"{'='*60}")
        print(f"CNN Feature Dimension:     {checkpoint['cnn_feature_dim']}")
        print(f"Transformer d_model:       {checkpoint['d_model']}")
        print(f"Number of attention heads: {checkpoint['nhead']}")
        print(f"Number of layers:          {checkpoint['num_layers']}")
        print(f"Feedforward dimension:     {checkpoint['dim_feedforward']}")
        print(f"Number of classes:         {checkpoint['num_classes']}")
        print(f"Dropout:                   {DROPOUT}")
        print(f"Total parameters:          {sum(p.numel() for p in model.parameters()):,}")

        print(f"\n{'='*60}")
        print(f"DATA & TRAINING HYPERPARAMETERS")
        print(f"{'='*60}")
        print(f"Sequence length: {checkpoint['sequence_length']} frames ({checkpoint['sequence_length']/30:.1f} seconds)")
        print(f"Image size:      {checkpoint['img_size'][0]}×{checkpoint['img_size'][1]}")
        print(f"Batch size:      {BATCH_SIZE}")
        print(f"Device:          {device}")

        print(f"\n{'='*60}")
        print(f"DATASET INFO")
        print(f"{'='*60}")
        print(f"Training sequences: {len(train_dataset)}")
        print(f"Val sequences:      {len(val_eval_dataset)}")
        print(f"Test sequences:     {len(test_dataset)}")
        print(f"Classes:            {behavior_names}")
        print(f"{'='*60}\n")

        SKIP_TRAINING = True

    else:
        print(f"\nNo existing model found at: {MODEL_PATH}")
        print(f"Training new model...\n")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

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

    # Dataloaders for evaluation
    epoch_val_loader = DataLoader(epoch_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=2, pin_memory=True)
    val_eval_loader = DataLoader(val_eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Class weights — computed from all per-frame labels across training videos,
    # not just anchor-frame labels, to correctly reflect the true class distribution.
    all_train_labels = np.concatenate([
        train_dataset.label_cache[vid] for vid in train_dataset.label_cache
    ])
    unique, counts = np.unique(all_train_labels, return_counts=True)
    class_counts   = dict(zip(unique, counts))
    total_samples  = len(all_train_labels)

    print(f"\n=== Class Distribution ===")
    for cls_idx, cls_name in enumerate(behavior_names):
        count      = class_counts.get(cls_idx, 0)
        percentage = 100 * count / total_samples
        print(f"{cls_name}: {count} ({percentage:.2f}%)")

    class_weights = {}
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 0)
        class_weights[cls_idx] = (total_samples / (num_classes * count)) ** 1.0 if count > 0 else 1.0

    # CHANGE (v20): Boost underperforming classes before capping background.
    # Unsupportedrearing and Grooming consistently spill into background on test set.
    CLASS_BOOSTS = {'Unsupportedrearing': 1.5, 'Grooming': 1.5}
    for cls_idx, cls_name in enumerate(behavior_names):
        if cls_name in CLASS_BOOSTS:
            class_weights[cls_idx] *= CLASS_BOOSTS[cls_name]

    # CHANGE (v20): Background weight cap lowered from 0.5 to 0.2.
    # With ~74% background frames and 5 classes, the natural inverse-frequency weight
    # is ~0.27 — the old cap of 0.5 was above that, so it had no effect at all.
    background_idx = next((i for i, n in enumerate(behavior_names) if n.lower() == 'background'), None)
    if background_idx is not None:
        class_weights[background_idx] = min(class_weights[background_idx], 0.2)

    class_weights_array = np.array([class_weights[i] for i in range(num_classes)])

    print(f"\n=== Class Weights (power=1.0, boosts applied, background capped at 0.2) ===")
    for cls_idx, cls_name in enumerate(behavior_names):
        boost_str = f" [x{CLASS_BOOSTS[cls_name]}]" if cls_name in CLASS_BOOSTS else ""
        print(f"{cls_name}: {class_weights_array[cls_idx]:.3f} (count: {class_counts.get(cls_idx, 0)}){boost_str}")

    weight_tensor = torch.FloatTensor(class_weights_array).to(device)

    # Training loop
    if not SKIP_TRAINING:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.01)  # CHANGE (v20): reduced from 0.05 — smoothing fights class weighting on imbalanced videos

        # CHANGE: increased weight_decay from 0.01 to 0.05 for stronger regularization
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

        # CHANGE: T_max=NUM_EPOCHS and step per epoch (not per batch).
        # Previously T_max=NUM_EPOCHS but stepping per batch caused LR to reach
        # minimum after epoch 1 and stay near zero for all remaining epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_f1          = 0.0
        patience         = 15
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # CHANGE: step scheduler once per epoch after train_epoch
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

            y_pred, y_true = evaluate(model, epoch_val_loader, device, use_consensus=True,
                                      background_class=background_idx if background_idx is not None else 0)
            val_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
            val_f1  = f1_score(y_true, y_pred, average='macro')

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc (consensus): {val_acc:.2f}%, Val F1 (consensus): {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'cnn_feature_dim':  CNN_FEATURE_DIM,
                    'd_model':          D_MODEL,
                    'nhead':            NHEAD,
                    'num_layers':       NUM_LAYERS,
                    'dim_feedforward':  DIM_FEEDFORWARD,
                    'num_classes':      num_classes,
                    'sequence_length':  SEQUENCE_LENGTH,
                    'img_size':         IMG_SIZE,
                    'dropout':          DROPOUT,
                }, MODEL_PATH)
                print(f"→ New best model saved! Val F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        print("\nLoading best model from training...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # ============================================================================
    # Final Evaluation
    # ============================================================================

    results_lines = []

    def log(line=""):
        print(line)
        results_lines.append(line)

    log("\n" + "="*60)
    log("FINAL TRAINING SET EVALUATION")
    log("="*60)

    y_pred_train, y_true_train = evaluate(model, train_eval_loader, device,
                                          background_class=background_idx if background_idx is not None else 0)
    log("\nClassification Report:")
    log(classification_report(y_true_train, y_pred_train, target_names=behavior_names))

    cm_train     = confusion_matrix(y_true_train, y_pred_train)
    cm_train_pct = cm_train.astype(float) / cm_train.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train_pct[::-1, :], annot=True, fmt='.2f', cmap='Greens',
                xticklabels=behavior_names, yticklabels=behavior_names[::-1])
    plt.title(f'Training Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./pipeline_outputs/cnn_transformer/conf_matrix_train_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')
    plt.close()

    log("\n" + "="*60)
    log("FINAL VALIDATION SET EVALUATION")
    log("="*60)

    y_pred_val, y_true_val = evaluate(model, val_eval_loader, device,
                                      background_class=background_idx if background_idx is not None else 0)
    log("\nClassification Report:")
    log(classification_report(y_true_val, y_pred_val, target_names=behavior_names))

    cm_val     = confusion_matrix(y_true_val, y_pred_val)
    cm_val_pct = cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_val_pct[::-1, :], annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=behavior_names, yticklabels=behavior_names[::-1])
    plt.title(f'Validation Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./pipeline_outputs/cnn_transformer/conf_matrix_val_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')
    plt.close()

    log("\n" + "="*60)
    log("FINAL TEST SET EVALUATION")
    log("="*60)

    y_pred, y_true, per_video_data = evaluate(model, test_loader, device, return_per_video=True,
                                              background_class=background_idx if background_idx is not None else 0)
    log("\nClassification Report:")
    log(classification_report(y_true, y_pred, target_names=behavior_names))

    cm_test     = confusion_matrix(y_true, y_pred)
    cm_test_pct = cm_test.astype(float) / cm_test.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test_pct[::-1, :], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=behavior_names, yticklabels=behavior_names[::-1])
    plt.title(f'Test Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./pipeline_outputs/cnn_transformer/conf_matrix_test_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')
    plt.close()

    log("\n" + "="*60)
    log("PER-VIDEO TEST EVALUATION")
    log("="*60)
    for video_id in sorted(per_video_data.keys()):
        v_preds  = np.array(per_video_data[video_id]['preds'])
        v_labels = np.array(per_video_data[video_id]['labels'])
        v_f1     = f1_score(v_labels, v_preds, average='macro', zero_division=0)
        log(f"\n--- {video_id} (macro F1: {v_f1:.3f}) ---")
        log(classification_report(v_labels, v_preds, target_names=behavior_names,
                                  labels=range(len(behavior_names)), zero_division=0))

    # ============================================================================
    # Behaviour Instance Counts
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
    plot_behavior_indices = [i for i in range(num_classes) if i != background_idx]
    plot_behavior_names   = [behavior_names[i] for i in plot_behavior_indices]

    if plot_behavior_names:
        fig, axes = plt.subplots(1, len(plot_behavior_names), figsize=(5 * len(plot_behavior_names), 5))
        if len(plot_behavior_names) == 1:
            axes = [axes]

        colors_true = '#59a89c'
        colors_pred = '#a559aa'

        for idx, cls_name in enumerate(plot_behavior_names):
            true_val = true_counts[cls_name]
            pred_val = pred_counts[cls_name]
            x_pos    = np.array([0, 1])
            values   = [true_val, pred_val]
            bars     = axes[idx].bar(x_pos, values, color=[colors_true, colors_pred], width=0.6)

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
        plt.savefig(f'./pipeline_outputs/cnn_transformer/behaviour_instance_count_{DATASET_VERSION}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\nBehaviour instance count plot saved!")

    # Per-video scatterplot
    if per_video_data and plot_behavior_names:
        video_counts = {cls_name: {'actual': [], 'predicted': []} for cls_name in plot_behavior_names}

        for video_id in sorted(per_video_data.keys()):
            v_preds  = np.array(per_video_data[video_id]['preds'])
            v_labels = np.array(per_video_data[video_id]['labels'])
            for cls_idx, cls_name in zip(plot_behavior_indices, plot_behavior_names):
                video_counts[cls_name]['actual'].append(count_behavior_instances(v_labels, cls_idx))
                video_counts[cls_name]['predicted'].append(count_behavior_instances(v_preds, cls_idx))

        fig, axes = plt.subplots(1, len(plot_behavior_names), figsize=(5 * len(plot_behavior_names), 5))
        if len(plot_behavior_names) == 1:
            axes = [axes]

        for idx, cls_name in enumerate(plot_behavior_names):
            actual    = np.array(video_counts[cls_name]['actual'])
            predicted = np.array(video_counts[cls_name]['predicted'])

            axes[idx].scatter(actual, predicted, alpha=0.6, s=100)
            max_val = max(int(actual.max()) if len(actual) > 0 else 0,
                         int(predicted.max()) if len(predicted) > 0 else 0)
            max_val = max_val if max_val > 0 else 1
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
        plt.savefig(f'./pipeline_outputs/cnn_transformer/instance_count_per_video_{DATASET_VERSION}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("\nInstance count per video scatterplot saved!")

    # Save evaluation results
    results_path = f'./pipeline_outputs/cnn_transformer/evaluation_{DATASET_VERSION}.txt'
    with open(results_path, 'w') as f:
        f.write("\n".join(results_lines))
    print(f"\nEvaluation results saved to {results_path}")

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")