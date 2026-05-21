"""
CNN-Transformer Prediction Script (no labels)

Loads a trained CNN-Transformer model, runs inference on all videos found in
VIDEO_FOLDER, and saves per-frame predicted labels as one CSV per video.

Output CSV format matches the training label format:
    columns: [behavior_1, behavior_2, ...] (one-hot encoded)
    index:   frame index
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import math
import time
from tqdm import tqdm
import joblib
from collections import defaultdict

_start_time = time.time()


# ============================================================================
# Model Architecture (must match training)
# ============================================================================

class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv_block(x))


class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 512, res_depth: int = 4, dropout: float = 0.3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.res_blocks_1 = nn.Sequential(*[ResBlock2D(48) for _ in range(res_depth)])
        self.transition_1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.res_blocks_2 = nn.Sequential(*[ResBlock2D(64) for _ in range(res_depth)])
        self.transition_2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(80),
            nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.res_blocks_3 = nn.Sequential(*[ResBlock2D(80) for _ in range(res_depth)])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 9 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks_1(x)
        x = self.transition_1(x)
        x = self.res_blocks_2(x)
        x = self.transition_2(x)
        x = self.res_blocks_3(x)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(self, cnn_feature_dim=512, d_model=512, nhead=8,
                 num_layers=2, num_classes=5, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(feature_dim=cnn_feature_dim, dropout=dropout)
        self.feature_projection = nn.Linear(cnn_feature_dim, d_model) if cnn_feature_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
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
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        x = self.feature_projection(cnn_features)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x)


# ============================================================================
# Dataset (no labels)
# ============================================================================

class VideoSequenceDatasetNoLabels(Dataset):
    """Sequences from videos with no ground-truth labels."""

    def __init__(self, video_folder, video_ids, sequence_length=30,
                 stride=5, img_size=(76, 142)):
        self.video_folder = video_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.sequence_info = []   # list of (video_id, ext, start_frame)

        print(f"Indexing sequences from {len(video_ids)} videos...")
        for video_id, ext in tqdm(video_ids, desc="Indexing videos"):
            video_path = os.path.join(video_folder, f"{video_id}{ext}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start_idx in range(0, total_frames - sequence_length + 1, stride):
                self.sequence_info.append((video_id, ext, start_idx))

        print(f"Indexed {len(self.sequence_info)} sequences")

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        video_id, ext, start_frame = self.sequence_info[idx]
        video_path = os.path.join(self.video_folder, f"{video_id}{ext}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(self.img_size[::-1], dtype=np.uint8))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.img_size)
                frames.append(gray)
        cap.release()

        frames = np.array(frames, dtype=np.float32)
        frames = -(frames / 255.0 - 0.5)
        frames = frames[:, np.newaxis, :, :]
        return torch.FloatTensor(frames)


# ============================================================================
# Prediction
# ============================================================================

def predict_per_video(model, dataloader, device, behavior_names):
    """
    Run inference and return per-video frame predictions using consensus voting.
    Returns dict: {video_id: (sorted_frame_indices, np.array of predicted class indices)}
    """
    model.eval()

    frame_predictions = defaultdict(list)

    with torch.no_grad():
        seq_idx_offset = 0
        for batch_X in tqdm(dataloader, desc="Predicting"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=2).cpu().numpy()

            actual_batch_size = batch_X.shape[0]
            for b in range(actual_batch_size):
                seq_idx = seq_idx_offset + b
                if seq_idx >= len(dataloader.dataset):
                    continue
                video_id, _ext, start_frame = dataloader.dataset.sequence_info[seq_idx]
                for frame_offset in range(dataloader.dataset.sequence_length):
                    key = (video_id, start_frame + frame_offset)
                    frame_predictions[key].append(probs[b, frame_offset])

            seq_idx_offset += actual_batch_size

    # Consensus vote per frame
    per_video_preds = defaultdict(dict)
    for (video_id, frame_idx), preds in frame_predictions.items():
        per_video_preds[video_id][frame_idx] = np.argmax(np.sum(preds, axis=0))

    result = {}
    for video_id, frame_dict in per_video_preds.items():
        sorted_frames = sorted(frame_dict.keys())
        preds = np.array([frame_dict[f] for f in sorted_frames])
        result[video_id] = (sorted_frames, preds)

    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    # ---- Configuration ----
    MODEL_PATH         = "./output/cnn_transformer/CNN_Transformer_lite_v24.pth"
    LABEL_ENCODER_PATH = "./output/cnn_transformer/label_encoder_lite_v24.pkl"
    VIDEO_FOLDER       = "./data/rotated_videos_maria"
    OUTPUT_FOLDER      = "./output/cnn_transformer/predictions_maria"

    SEQUENCE_LENGTH = 30
    EVAL_STRIDE     = 5
    IMG_SIZE        = (76, 142)
    BATCH_SIZE      = 32
    DROPOUT         = 0.3
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
    # -----------------------

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label encoder
    behavior_names = joblib.load(LABEL_ENCODER_PATH)
    print(f"Classes: {behavior_names}")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = CNNTransformerClassifier(
        cnn_feature_dim=checkpoint['cnn_feature_dim'],
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes'],
        dim_feedforward=checkpoint['dim_feedforward'],
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded.")

    # Auto-discover all video files in the folder
    video_ids = []
    for fname in sorted(os.listdir(VIDEO_FOLDER)):
        name, ext = os.path.splitext(fname)
        if ext.lower() in VIDEO_EXTENSIONS:
            video_ids.append((name, ext))

    if not video_ids:
        print(f"No video files found in {VIDEO_FOLDER}. Exiting.")
        exit(0)

    print(f"Found {len(video_ids)} video(s): {[v for v, _ in video_ids]}")

    # Build dataset & loader
    dataset = VideoSequenceDatasetNoLabels(
        VIDEO_FOLDER, video_ids,
        SEQUENCE_LENGTH, EVAL_STRIDE, IMG_SIZE
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Predict
    per_video = predict_per_video(model, loader, device, behavior_names)

    # Save one CSV per video
    for video_id, (frame_indices, preds) in per_video.items():
        one_hot = np.zeros((len(preds), len(behavior_names)), dtype=int)
        for i, cls_idx in enumerate(preds):
            one_hot[i, cls_idx] = 1

        df = pd.DataFrame(one_hot, columns=behavior_names, index=frame_indices)
        df.index.name = "frame"

        out_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")
        df.to_csv(out_path)
        print(f"Saved: {out_path}  ({len(df)} frames)")

    elapsed = time.time() - _start_time
    print(f"\nDone. Predictions saved to: {OUTPUT_FOLDER}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
