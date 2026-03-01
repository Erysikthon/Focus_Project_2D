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
DATASET_VERSION = "Transformer_hist_2"

X_path = f"./pipeline_saved_processes/dataframes/X_hist.csv"
X_filtered_path = f"./pipeline_saved_processes/dataframes/X_hist_filtered.csv"
y_path = f"./pipeline_saved_processes/dataframes/y_hist.csv"
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

    collection_path = "./pipeline_inputs/collection"
    fps = 30
    rescale_points = ("tr", "tl")
    rescale_distance = 0.64
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

    # Likelihood filter
    tracking_collection.each.filter_likelihood(filter_threshold)

    # Rescale (2D only - x, y)
    tracking_collection.each.rescale_by_known_distance(rescale_points[0], rescale_points[1], rescale_distance, dims=("x", "y"))

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

                               embedding_length=list(range(-15, 16, 3))
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


# Transformer Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dim_feedforward=512, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU works better than ReLU for transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Enhanced classification head with residual connection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
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
        # x shape: (batch, seq_len, input_size)

        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
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


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return np.array(all_preds), np.array(all_labels)


if not os.path.isfile(model_path):

    # Option 1: Manually define test video IDs (set to None to use random split)
    manual_test_video_ids = ["T2","T4","T13","MBT1-M2","MBT1-M7","MBT1-M10"]  # Example: ['video1', 'video2', 'video3']

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

    # Calculate class weights
    y_train_flat = y_train_encoded.values.ravel()
    unique, counts = np.unique(y_train_flat, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Class distribution in training: {class_counts}")

    total_samples = len(y_train_flat)
    n_classes = len(unique)
    class_weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
    print(f"Class weights: {class_weights}")

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

    # Increase batch size for GPU efficiency
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model - prioritize CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    input_size = X_train.shape[1]
    d_model = 256        # Increased embedding dimension for more capacity
    nhead = 8            # Number of attention heads (must divide d_model)
    num_layers = 6       # Increased layers for more model capacity
    dim_feedforward = 1024  # Increased feedforward dimension
    num_classes = len(unique)

    model = TransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_feedforward,
        dropout=0.3
    ).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted loss for imbalanced classes
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)  # Add label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Higher LR, stronger regularization
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # Better scheduler

    # Training loop
    num_epochs = 150
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement to reset patience

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        y_pred, y_true = evaluate(model, test_loader, device)
        test_acc = 100 * np.sum(y_pred == y_true) / len(y_true)
        test_f1 = f1_score(y_true, y_pred, average='macro')

        scheduler.step()  # CosineAnnealingWarmRestarts doesn't need metric

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}")

        # Early stopping with minimum delta
        if test_f1 > best_f1 + min_delta:
            best_f1 = test_f1
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

else:
    # Load existing model
    import joblib

    checkpoint = torch.load(model_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        dropout=0.3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

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

# Training set evaluation
print("\n=== Training Set Evaluation ===")
y_pred_train, y_true_train = evaluate(model, train_loader, device)

train_acc = 100 * np.sum(y_pred_train == y_true_train) / len(y_true_train)
train_f1 = f1_score(y_true_train, y_pred_train, average='macro')

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Train F1 Score (macro): {train_f1:.4f}")
print("\nTraining Classification Report:")
print(classification_report(y_true_train, y_pred_train, target_names=label_encoder.classes_))

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
