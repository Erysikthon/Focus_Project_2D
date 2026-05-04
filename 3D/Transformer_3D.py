import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pipeline_code.generate_features_ai import triangulate
from pipeline_code.generate_features_ai import features
from pipeline_code.generate_labels import labels
from pipeline_code.fix_frames import drop_non_analyzed_videos
from pipeline_code.fix_frames import drop_last_frame
from pipeline_code.fix_frames import drop_nas
from pipeline_code.filter_and_preprocess import reduce_bits
from pipeline_code.filter_and_preprocess import collinearity_filter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
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
DATASET_VERSION = "Transformer_3D_v1"

X_path = f"./pipeline_saved_processes/dataframes/X_3D.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_3D_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_3D.csv"
model_path = f"pipeline_saved_processes/models/Transformer_{DATASET_VERSION}.pth"
scaler_path = f"pipeline_saved_processes/models/scaler_{DATASET_VERSION}.pkl"
label_encoder_path = f"pipeline_saved_processes/models/label_encoder_{DATASET_VERSION}.pkl"

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

                               is_point_recognized=("nose",),

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

def _check_video_ids(X, y, path):
    x_ids = set(X.index.get_level_values("video_id").unique())
    y_ids = set(y.index.get_level_values("video_id").unique())
    if x_ids != y_ids:
        missing = y_ids - x_ids
        raise RuntimeError(
            f"Cached {path} is incomplete: missing video_ids {sorted(missing)}. "
            f"Delete it to regenerate."
        )

_check_video_ids(X, y, X_path)

# Apply pure collinearity filtering (no target variable used)
if os.path.isfile(X_filtered_path):
    print("Loading filtered X...")
    X = pd.read_csv(X_filtered_path, index_col=["video_id", "frame"])
    _check_video_ids(X, y, X_filtered_path)
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


# OFT dataset: 20 videos (collection 1-21, no 5) — 14 train / 3 val / 3 test (70/15/15)
_val_ids  = [7, 14, 20]
_test_ids = [4, 11, 17]

if not os.path.isfile(model_path):

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
    num_classes = n_classes
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

    # Hardcoded best parameters from grid search
    SEQUENCE_LENGTH = 30
    STRIDE          = 10
    d_model         = 512
    num_layers      = 3
    nhead           = 8
    dim_feedforward = 1024
    dropout         = 0.3
    learning_rate   = 0.0003
    batch_size      = 512

    # Create sequence datasets with selected hyperparameters
    print(f"\nCreating final datasets with length {SEQUENCE_LENGTH} and stride {STRIDE}...")
    train_dataset = SequenceDataset(X_train_scaled, y_train_encoded, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    val_dataset   = SequenceDataset(X_val_scaled,   y_val_encoded,   sequence_length=SEQUENCE_LENGTH, stride=STRIDE)
    test_dataset  = SequenceDataset(X_test_scaled,  y_test_encoded,  sequence_length=SEQUENCE_LENGTH, stride=STRIDE)

    print(f"Total training sequences: {len(train_dataset)}, val sequences: {len(val_dataset)}, test sequences: {len(test_dataset)}")
    print(f"Class distribution in training sequences: {Counter(train_dataset.labels.numpy().flatten())}")

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

    num_epochs = 150
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_f1 = 0.0
    patience = 20  # Increased patience for larger model
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement to reset patience

    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout,  # saved so the loaded model matches exactly
        'num_classes': num_classes,
        'sequence_length': SEQUENCE_LENGTH,
        'class_weights': class_weights,
        'train_videos': X_train.index.get_level_values('video_id').unique().tolist(),
        'val_videos': X_val.index.get_level_values('video_id').unique().tolist(),
        'test_videos': X_test.index.get_level_values('video_id').unique().tolist()
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

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
            checkpoint_data['model_state_dict'] = model.state_dict()
            torch.save(checkpoint_data, model_path)
            print(f"  → New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save final model if no checkpoint was saved during training
    if not os.path.isfile(model_path):
        print("Warning: val F1 never improved during training. Saving final model as fallback.")
        checkpoint_data['model_state_dict'] = model.state_dict()
        torch.save(checkpoint_data, model_path)

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
        dropout=checkpoint.get('dropout', 0.3)
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

    # Load scaler and scale videos
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

results_lines = []

def log(line=""):
    print(line)
    results_lines.append(line)

# Training set evaluation
log("\n" + "="*60)
log("FINAL TRAINING SET EVALUATION")
log("="*60)
y_pred_train, y_true_train = evaluate(model, train_loader, device)

train_acc = 100 * np.sum(y_pred_train == y_true_train) / len(y_true_train)
train_f1  = f1_score(y_true_train, y_pred_train, average='macro')

log(f"Train Accuracy: {train_acc:.2f}%")
log(f"Train F1 Score (macro): {train_f1:.4f}")
log("\nClassification Report:")
log(classification_report(y_true_train, y_pred_train, target_names=label_encoder.classes_))

# Val set evaluation
log("\n" + "="*60)
log("FINAL VALIDATION SET EVALUATION")
log("="*60)
y_pred_val, y_true_val = evaluate(model, val_loader, device)

val_acc = 100 * np.sum(y_pred_val == y_true_val) / len(y_true_val)
val_f1  = f1_score(y_true_val, y_pred_val, average='macro')

log(f"Val Accuracy: {val_acc:.2f}%")
log(f"Val F1 Score (macro): {val_f1:.4f}")
log("\nClassification Report:")
log(classification_report(y_true_val, y_pred_val, target_names=label_encoder.classes_))

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
log("\n" + "="*60)
log("FINAL TEST SET EVALUATION")
log("="*60)
y_pred, y_true = evaluate(model, test_loader, device)

test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
test_f1  = f1_score(y_true, y_pred, average='macro')

log(f"Test Accuracy: {test_acc:.2f}%")
log(f"Test F1 Score (macro): {test_f1:.4f}")
log("\nClassification Report:")
log(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

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

results_path = f'pipeline_outputs/evaluation_{DATASET_VERSION}.txt'
with open(results_path, 'w') as f:
    f.write("\n".join(results_lines))
print(f"\nEvaluation results saved to {results_path}")

print(f"\nTotal time: {time.time() - start:.2f} seconds")
