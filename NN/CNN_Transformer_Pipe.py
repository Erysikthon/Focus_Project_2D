"""
CNN-Transformer Pipeline for Mouse Behavior Classification from Raw Video

Architecture:
1. CNN Feature Extractor: Extracts spatial features from individual video frames
2. Transformer: Captures temporal dependencies across frame sequences
3. Classification Head: Predicts behavior labels

Input: Raw video frames from rotated_videos folder
Output: Behavior classification per frame/sequence
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib


# ============================================================================
# Helper Functions
# ============================================================================


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
        return x + self.conv_block(x)


class CNNFeatureExtractor(nn.Module):
    """
    CNN to extract spatial features from individual frames
    Input: (batch, 1, H, W) - grayscale frames
    Output: (batch, feature_dim) - feature vector per frame
    """
    def __init__(self, feature_dim=512):
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ResBlock population 1
        self.res_blocks_1 = nn.Sequential(*[ResBlock2D(48) for _ in range(3)])

        # Transition layer
        self.transition_1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ResBlock population 2
        self.res_blocks_2 = nn.Sequential(*[ResBlock2D(64) for _ in range(3)])

        # Transition layer
        self.transition_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )

        # Feature projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
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
    Combined CNN-Transformer for video behavior classification

    Architecture:
    1. CNN extracts features from each frame independently
    2. Transformer processes temporal sequence of CNN features
    3. Classification head predicts behavior
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

        # CLS token for classification (like BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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

        # Classification head
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
        returns: (batch, num_classes) - classification logits
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

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len+1, d_model)

        # Use CLS token for classification
        cls_output = x[:, 0]  # (batch, d_model)

        # Classification
        logits = self.classifier(cls_output)  # (batch, num_classes)

        return logits


# ============================================================================
# Video Dataset
# ============================================================================

class VideoSequenceDataset(Dataset):
    """
    Lazy-loading dataset for video sequences (loads frames on-demand to save memory)
    Supports multi-scale temporal windows for better handling of variable-duration behaviors
    """
    def __init__(self, video_folder, label_folder, video_ids, sequence_length=30,
                 stride=10, img_size=(128, 128), sequence_lengths=None):
        """
        Args:
            video_folder: Path to folder containing .mp4 files
            label_folder: Path to folder containing label CSV files
            video_ids: List of video IDs (filenames without extension)
            sequence_length: Default number of frames per sequence (used if sequence_lengths is None)
            stride: Step size between sequences
            img_size: (height, width) to resize frames
            sequence_lengths: List of sequence lengths for multi-scale training (e.g., [15, 30, 45])
                             If None, uses fixed sequence_length
        """
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.sequence_lengths = sequence_lengths if sequence_lengths is not None else [sequence_length]
        self.use_multiscale = sequence_lengths is not None

        # Store metadata about sequences (not the actual frames)
        self.sequence_info = []  # List of (video_id, start_frame, label, seq_len)
        self.labels = []

        print(f"Indexing sequences from {len(video_ids)} videos...")
        self._index_sequences(video_ids)

    def _index_sequences(self, video_ids):
        """Create index of sequences without loading video frames"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
            label_path = os.path.join(self.label_folder, f"{video_id}_labels.csv")

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
                behavior_columns = [col for col in labels_df.columns if col not in ['Unnamed: 0', 'frame']]
                if len(behavior_columns) > 0:
                    video_labels = labels_df[behavior_columns].values.argmax(axis=1)
                else:
                    print(f"Warning: Invalid label format for {video_id}")
                    continue
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
            # Use maximum sequence length for indexing to ensure all frames are available
            max_seq_len = max(self.sequence_lengths)

            for start_idx in range(0, min(total_frames, len(video_labels)) - max_seq_len + 1, self.stride):
                # Use primary sequence length for label extraction
                end_idx = start_idx + self.sequence_length
                label = video_labels[end_idx - 1]

                # Randomly choose sequence length if using multi-scale
                if self.use_multiscale:
                    seq_len = np.random.choice(self.sequence_lengths)
                else:
                    seq_len = self.sequence_length

                self.sequence_info.append((video_id, start_idx, label, seq_len))
                self.labels.append(label)

        multiscale_msg = f" (multi-scale: {self.sequence_lengths})" if self.use_multiscale else ""
        print(f"Indexed {len(self.sequence_info)} sequences{multiscale_msg}")

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        """Load video frames on-demand"""
        # Handle both old format (3-tuple) and new format (4-tuple) for backward compatibility
        seq_info = self.sequence_info[idx]
        if len(seq_info) == 4:
            video_id, start_frame, label, seq_len = seq_info
        else:
            # Old format without seq_len
            video_id, start_frame, label = seq_info
            seq_len = self.sequence_length

        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        # Open video and load the specific sequence
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(seq_len):
            ret, frame = cap.read()
            if not ret:
                # If we can't read a frame, use a black frame
                frames.append(np.zeros(self.img_size, dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, self.img_size)
                frames.append(resized)

        cap.release()

        # Convert to numpy array
        frames = np.array(frames, dtype=np.float32)

        # Normalize all sequences to fixed length (30) for batch processing
        target_len = 30
        if len(frames) < target_len:
            # Pad with zeros (black frames)
            padding = np.zeros((target_len - len(frames), *self.img_size), dtype=np.float32)
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > target_len:
            # Sample frames uniformly to get target length
            indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
            frames = frames[indices]

        # Normalize to [0, 1]
        frames = frames / 255.0

        # Add channel dimension: (seq_len, H, W) -> (seq_len, 1, H, W)
        frames = frames[:, np.newaxis, :, :]

        return torch.FloatTensor(frames), label


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in tqdm(dataloader, desc="Training"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Configuration
    DATASET_VERSION = "CNN_Transformer_v4_multiscale_only"
    VIDEO_FOLDER = "./data/rotated_videos"
    LABEL_FOLDER = "./data/labels"
    MODEL_PATH = f"./output_cnn_transformer/CNN_Transformer_{DATASET_VERSION}.pth"
    SCALER_PATH = f"./output_cnn_transformer/scaler_{DATASET_VERSION}.pkl"
    LABEL_ENCODER_PATH = f"./output_cnn_transformer/label_encoder_{DATASET_VERSION}.pkl"

    # Hyperparameters (optimized for speed + temporal context)
    SEQUENCE_LENGTH = 30
    STRIDE = 10
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32  # Increased from 16 for faster training
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001

    CNN_FEATURE_DIM = 512
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get video IDs
    all_video_ids = [f.replace('.mp4', '') for f in os.listdir(VIDEO_FOLDER)
                     if f.endswith('.mp4')]

    # Manual train/test split (matching Transformer_Pipe_2D.py)
    manual_test_video_ids = ['3279_21min_behaviour_2023-01-19T12_57_29', '20231123_10min_OFT-BL_4028',
                             'BehavioralCamera2023-02-23T10_23_42_shorter', 'MBT1-M2', 'T2',
                             'MBT1-M7', 'T8', 'T4', 'BehavioralCamera2023-02-24T11_06_53_shorter', 'T1']

    # Use manually specified test videos
    test_video_ids = [vid for vid in manual_test_video_ids if vid in all_video_ids]
    train_video_ids = [vid for vid in all_video_ids if vid not in test_video_ids]

    print(f"Manual split: Total videos: {len(all_video_ids)}, Test videos: {len(test_video_ids)}, Train videos: {len(train_video_ids)}")
    print(f"Test video IDs: {test_video_ids}")

    # Create datasets
    print(f"\nCreating training dataset from {len(train_video_ids)} videos...")
    train_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, train_video_ids,
        SEQUENCE_LENGTH, STRIDE, IMG_SIZE,
        sequence_lengths=[15, 30, 45]  # Multi-scale: short (0.5s), medium (1s), long (1.5s)
    )
    print(f"Training dataset created: {len(train_dataset)} sequences")

    print(f"\nCreating test dataset from {len(test_video_ids)} videos...")
    test_dataset = VideoSequenceDataset(
        VIDEO_FOLDER, LABEL_FOLDER, test_video_ids,
        SEQUENCE_LENGTH, STRIDE, IMG_SIZE,
        sequence_lengths=None  # Fixed length for test (consistent evaluation)
    )
    print(f"Test dataset created: {len(test_dataset)} sequences")

    # Get behavior class names from first label file
    sample_label_path = os.path.join(LABEL_FOLDER, f"{all_video_ids[0]}_labels.csv")
    sample_df = pd.read_csv(sample_label_path)
    behavior_names = [col for col in sample_df.columns if col not in ['Unnamed: 0', 'frame']]

    num_classes = len(behavior_names)
    print(f"Classes: {behavior_names}")
    print(f"Number of classes: {num_classes}")

    # Convert labels to PyTorch tensors (they're already integer indices from argmax)
    train_dataset.labels = [int(label) for label in train_dataset.labels]
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

    # Create test dataloader (always needed for evaluation)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Create train evaluation dataloader (for final evaluation only, not training)
    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
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

    # Calculate class weights (power of 0.7 for moderate balancing)
    class_weights = {}
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 0)
        if count > 0:
            class_weights[cls_idx] = (total_samples / (n_classes * count)) ** 0.7
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
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_f1 = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)

            # Evaluate
            y_pred, y_true = evaluate(model, test_loader, device)
            test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
            test_f1 = f1_score(y_true, y_pred, average='macro')

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")

            # Save best model
            if test_f1 > best_f1:
                best_f1 = test_f1
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
                print(f"→ New best model saved! F1: {best_f1:.4f}")
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
    print("\n" + "="*60)
    print("FINAL TRAINING SET EVALUATION")
    print("="*60)

    y_pred_train, y_true_train = evaluate(model, train_eval_loader, device)
    print("\nClassification Report:")
    print(classification_report(y_true_train, y_pred_train, target_names=behavior_names))

    # Training confusion matrix
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                xticklabels=behavior_names,
                yticklabels=behavior_names)
    plt.title(f'Training Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./output_cnn_transformer/conf_matrix_train_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    y_pred, y_true = evaluate(model, test_loader, device)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=behavior_names))

    # Test confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=behavior_names,
                yticklabels=behavior_names)
    plt.title(f'Test Set Confusion Matrix - {DATASET_VERSION}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'./output_cnn_transformer/conf_matrix_test_{DATASET_VERSION}.png', dpi=300, bbox_inches='tight')

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
