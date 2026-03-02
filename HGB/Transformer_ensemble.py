"""
Ensemble predictions from multiple Transformer models
Combines hist_6 and hist_8 for improved performance
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
import joblib
from collections import Counter

# Import model architecture
import sys
sys.path.append('.')
from Transformer_Pipe_2D import TransformerClassifier, SequenceDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
X = pd.read_csv("./pipeline_saved_processes/dataframes/X_hist_filtered.csv", index_col=["video_id", "frame"])
y = pd.read_csv("./pipeline_saved_processes/dataframes/y_hist.csv", index_col=["video_id", "frame"])

# Model configurations
models_config = {
    'hist_6': {
        'path': 'pipeline_saved_processes/models/Transformer_Transformer_hist_6.pth',
        'scaler': 'pipeline_saved_processes/models/scaler_Transformer_hist_6.pkl',
        'label_encoder': 'pipeline_saved_processes/models/label_encoder_Transformer_hist_6.pkl'
    },
    'hist_8': {
        'path': 'pipeline_saved_processes/models/Transformer_Transformer_hist_8.pth',
        'scaler': 'pipeline_saved_processes/models/scaler_Transformer_hist_8.pkl',
        'label_encoder': 'pipeline_saved_processes/models/label_encoder_Transformer_hist_8.pkl'
    }
}

def load_model_and_predict(model_name, config):
    """Load a model and return its predictions"""
    print(f"\n=== Loading {model_name} ===")

    # Load checkpoint
    checkpoint = torch.load(config['path'], weights_only=False)

    # Load label encoder and scaler
    label_encoder = joblib.load(config['label_encoder'])
    scaler = joblib.load(config['scaler'])

    # Recreate test split
    test_video_ids = checkpoint['test_videos']
    X_test = X.loc[X.index.get_level_values('video_id').isin(test_video_ids)]
    y_test = y.loc[y.index.get_level_values('video_id').isin(test_video_ids)]

    # Encode labels
    y_test_encoded = pd.DataFrame(
        label_encoder.transform(y_test.values.ravel()),
        index=y_test.index,
        columns=[y_test.name] if isinstance(y_test, pd.Series) else y_test.columns
    )

    # Scale features
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    # Create dataset
    SEQUENCE_LENGTH = checkpoint['sequence_length']
    test_dataset = SequenceDataset(X_test_scaled, y_test_encoded, sequence_length=SEQUENCE_LENGTH, stride=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Load model
    model = TransformerClassifier(
        input_size=checkpoint['input_size'],
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        num_classes=checkpoint['num_classes'],
        dim_feedforward=checkpoint['dim_feedforward'],
        dropout=0.1 if model_name == 'hist_6' else 0.4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get predictions with probabilities
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)

    # Individual model performance
    preds = np.argmax(all_probs, axis=1)
    f1 = f1_score(all_labels, preds, average='macro')
    print(f"{model_name} Test F1: {f1:.4f}")

    return all_probs, all_labels, label_encoder

# Load predictions from both models
print("Loading models and generating predictions...")
probs_6, labels_6, le_6 = load_model_and_predict('hist_6', models_config['hist_6'])
probs_8, labels_8, le_8 = load_model_and_predict('hist_8', models_config['hist_8'])

# Verify labels match
assert np.array_equal(labels_6, labels_8), "Labels don't match between models!"
assert np.array_equal(le_6.classes_, le_8.classes_), "Label encoders don't match!"

# Ensemble: Average probabilities
print("\n=== Ensemble Predictions ===")
ensemble_probs = (probs_6 + probs_8) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Evaluate ensemble
ensemble_f1 = f1_score(labels_6, ensemble_preds, average='macro')
ensemble_acc = 100 * np.sum(ensemble_preds == labels_6) / len(labels_6)

print(f"\nEnsemble Test Accuracy: {ensemble_acc:.2f}%")
print(f"Ensemble Test F1 Score (macro): {ensemble_f1:.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(labels_6, ensemble_preds, target_names=le_6.classes_))

# Weighted ensemble (try different weights)
print("\n=== Weighted Ensemble (70% hist_8, 30% hist_6) ===")
weighted_probs = 0.7 * probs_8 + 0.3 * probs_6
weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_f1 = f1_score(labels_6, weighted_preds, average='macro')
weighted_acc = 100 * np.sum(weighted_preds == labels_6) / len(labels_6)

print(f"Weighted Test Accuracy: {weighted_acc:.2f}%")
print(f"Weighted Test F1 Score (macro): {weighted_f1:.4f}")
print("\nWeighted Classification Report:")
print(classification_report(labels_6, weighted_preds, target_names=le_6.classes_))
