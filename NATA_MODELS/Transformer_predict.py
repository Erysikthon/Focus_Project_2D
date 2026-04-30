"""
Transformer Prediction Script

Loads a trained Transformer model directly from a .pth checkpoint,
runs inference on the test set, and saves per-frame predicted labels as
one CSV per video.

The .pth file is produced by Transformer_Pipe_2D_lite.py:
    pipeline_saved_processes/models/Transformer_<VERSION>.pth

Scaler and label encoder are loaded from the same folder using the version
tag derived from the model filename.

Output CSV format matches the training label format:
    columns: [behavior_1, behavior_2, ...] (one-hot encoded)
    index:   frame index
"""

import os
import math
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "./pipeline_saved_processes/models/Transformer_Transformer_v2.pth"

X_FILTERED_PATH = "./pipeline_saved_processes/dataframes/X_filtered.csv"
Y_PATH          = "./pipeline_saved_processes/dataframes/y.csv"
OUTPUT_FOLDER   = "./pipeline_outputs/Transformer/predictions"

EVAL_STRIDE = 5
BATCH_SIZE  = 64

# Override test videos (set to None to use the list saved in the checkpoint)
MANUAL_TEST_VIDEO_IDS = None


# ============================================================================
# Model Architecture (must match training)
# ============================================================================

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


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x)


# ============================================================================
# Dataset
# ============================================================================

class SequenceDataset(Dataset):
    def __init__(self, X, video_ids, sequence_length, stride):
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_info = []

        for video_id in video_ids:
            if video_id not in X.index.get_level_values("video_id"):
                continue
            video_X = X.loc[video_id].values
            for start in range(0, len(video_X) - sequence_length + 1, stride):
                self.sequences.append(video_X[start:start + sequence_length])
                self.sequence_info.append((video_id, start))

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        print(f"Built {len(self.sequences)} sequences (stride={stride})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================================
# Prediction
# ============================================================================

def predict_per_video(model, dataloader, device):
    model.eval()
    frame_probs = defaultdict(list)

    with torch.no_grad():
        seq_offset = 0
        for batch_X in tqdm(dataloader, desc="Predicting"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)                          # (B, seq_len, num_classes)
            probs = torch.softmax(outputs, dim=2).cpu().numpy()

            for b in range(batch_X.shape[0]):
                seq_idx = seq_offset + b
                if seq_idx >= len(dataloader.dataset):
                    break
                video_id, start_frame = dataloader.dataset.sequence_info[seq_idx]
                for offset in range(dataloader.dataset.sequence_length):
                    frame_probs[(video_id, start_frame + offset)].append(probs[b, offset])

            seq_offset += batch_X.shape[0]

    # Consensus vote: sum softmax probabilities across overlapping sequences
    per_video = defaultdict(dict)
    for (video_id, frame_idx), prob_list in frame_probs.items():
        per_video[video_id][frame_idx] = np.argmax(np.sum(prob_list, axis=0))

    result = {}
    for video_id, frame_dict in per_video.items():
        sorted_frames = sorted(frame_dict.keys())
        result[video_id] = (sorted_frames, np.array([frame_dict[f] for f in sorted_frames]))

    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    # ---- Device ----
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ---- Load model checkpoint ----
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Derive version tag and companion file paths from model filename
    model_basename = os.path.basename(MODEL_PATH)                       # Transformer_<VERSION>.pth
    dataset_version = model_basename.replace("Transformer_", "").replace(".pth", "")
    models_dir = os.path.dirname(MODEL_PATH)

    scaler_path        = os.path.join(models_dir, f"scaler_Transformer_{dataset_version}.pkl")
    label_encoder_path = os.path.join(models_dir, f"label_encoder_Transformer_{dataset_version}.pkl")

    # ---- Load label encoder & scaler ----
    label_encoder = joblib.load(label_encoder_path)
    behavior_names = list(label_encoder.classes_)
    print(f"Classes: {behavior_names}")

    scaler = joblib.load(scaler_path)

    # ---- Reconstruct model ----
    model = TransformerClassifier(
        input_size     = checkpoint["input_size"],
        d_model        = checkpoint["d_model"],
        nhead          = checkpoint["nhead"],
        num_layers     = checkpoint["num_layers"],
        num_classes    = checkpoint["num_classes"],
        dim_feedforward= checkpoint["dim_feedforward"],
        dropout        = checkpoint.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded.")

    sequence_length = checkpoint["sequence_length"]

    # ---- Determine test videos ----
    if MANUAL_TEST_VIDEO_IDS is not None:
        test_video_ids = MANUAL_TEST_VIDEO_IDS
    elif "test_videos" in checkpoint:
        test_video_ids = checkpoint["test_videos"]
        print(f"Using test videos from checkpoint ({len(test_video_ids)} videos)")
    else:
        raise ValueError("No test_video_ids provided and checkpoint has no 'test_videos' key.")

    # ---- Load features ----
    print("Loading features...")
    X = pd.read_csv(X_FILTERED_PATH, index_col=["video_id", "frame"])

    all_video_ids  = X.index.get_level_values("video_id").unique().tolist()
    test_video_ids = [v for v in test_video_ids if v in all_video_ids]

    if MANUAL_TEST_VIDEO_IDS is not None:
        missing = set(MANUAL_TEST_VIDEO_IDS) - set(all_video_ids)
        if missing:
            print(f"Warning: test videos not found in data: {missing}")

    print(f"Test videos: {len(test_video_ids)}")

    # ---- Scale features ----
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        index=X.index,
        columns=X.columns,
    )

    # ---- Build dataset & loader ----
    dataset = SequenceDataset(X_scaled, test_video_ids, sequence_length, EVAL_STRIDE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Predict ----
    per_video = predict_per_video(model, loader, device)

    # ---- Save one CSV per video ----
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for video_id, (frame_indices, preds) in per_video.items():
        one_hot = np.zeros((len(preds), len(behavior_names)), dtype=int)
        for i, cls_idx in enumerate(preds):
            one_hot[i, cls_idx] = 1

        df = pd.DataFrame(one_hot, columns=behavior_names, index=frame_indices)
        df.index.name = "frame"

        out_path = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")
        df.to_csv(out_path)
        print(f"Saved: {out_path}  ({len(df)} frames)")

    print(f"\nDone. Predictions saved to: {OUTPUT_FOLDER}")
