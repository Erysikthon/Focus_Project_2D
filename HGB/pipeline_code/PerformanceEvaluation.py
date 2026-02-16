

########################################## ACCURACY & CLASSIFICATION REPORT ############################################
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import numpy as np


def count_behavior_instances(predictions, behavior_label):
    """
    Count the number of behavior instances by detecting transitions to the behavior.
    A new instance is counted each time the behavior starts (transition from another label to this label).

    Args:
        predictions: Array of predicted labels
        behavior_label: The specific behavior to count

    Returns:
        Number of instances of the behavior
    """

    # Create binary array: 1 where prediction matches behavior, 0 otherwise
    behavior_mask = (predictions == behavior_label).astype(int)

    # Detect transitions: count where diff > 0.5 (i.e., transition from 0 to 1)
    changes = np.diff(behavior_mask, prepend=0)
    instances = (changes > 0.5).sum()

    return instances

def smooth_predictions(predictions, min_frames):
    
    #Remove behavior prediction outliers shorter than min_frames.

       # predictions: Array of predicted labels
      #  min_frames: Minimum number of consecutive frames for a behavior to be valid

    #Returns:
      #  Smoothed predictions array
    
    if len(predictions) < min_frames:
        return predictions

    smoothed = predictions.copy()
    i = 0

    while i < len(smoothed):
        current_label = smoothed[i]
        # Find the length of the current behavior segment
        segment_start = i
        while i < len(smoothed) and smoothed[i] == current_label:
            i += 1
        segment_length = i - segment_start

        # If segment is shorter than min_frames, replace with neighboring behavior
        if segment_length < min_frames:
            # Determine replacement label from neighbors
            prev_label = smoothed[segment_start - 1] if segment_start > 0 else None
            next_label = smoothed[i] if i < len(smoothed) else None

            # Choose the label that appears in neighbors (prefer previous)
            if prev_label is not None and next_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label
            else:
                # Keep original if no neighbors available
                continue

            # Replace the short segment
            smoothed[segment_start:i] = replacement

    return smoothed

def evaluate_model(model : BaseEstimator, X_train : pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, min_frames=None, conf_matrix_path : str = "pipeline_outputs/conf_matrix.png" ):
    y_pred = model.predict(X_test)

    # Apply smoothing to remove outliers below 20 frames
    if min_frames is not None:
        y_pred = smooth_predictions(y_pred, min_frames)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #print(f"Cross-validation score: {cross_val_score(model, X_train, y_train, cv=5).mean():.4f}")

    print("\n=== Classification Report - TRAIN SET ===")
    y_train_pred = model.predict(X_train)
    if min_frames is not None:
        y_train_pred = smooth_predictions(y_train_pred, min_frames )
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(classification_report(y_train, y_train_pred))

    print("\n=== Classification Report - TEST SET ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    ################################### CONFUSION MATRIX ###############################################################




    # Get labels in the order used by confusion_matrix
    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Print absolute confusion matrix
    print("\n=== Confusion Matrix (Absolute Frames) ===")
    print("Labels order:", labels)
    print(cm)

    # Non-normalized Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',              # Format as integers
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Frame Count'}
    )
    plt.title('Confusion Matrix (Absolute Frames)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, labelpad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=15)
    plt.tight_layout()
    conf_matrix_abs_path = conf_matrix_path.replace('.png', '_absolute.png')
    plt.savefig(conf_matrix_abs_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Normalized confusion matrix
    print("\n=== Normalised Confusion Matrix ===")
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cmn)

    # Normalized Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cmn,
        annot=True,           # Show numbers in cells
        fmt='.2f',              # Format as percentages
        cmap='Blues',         # Color scheme
        xticklabels=labels,   # Label x-axis with class names in correct order
        yticklabels=labels,   # Label y-axis with class names in correct order
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, labelpad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=15)
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    ################################### BEHAVIOR INSTANCE COUNTS ###############################################################

    print("\n=== Behavior Instance Counts - TRAIN SET ===")
    behaviors_train = np.unique(y_train)

    print(f"{'Behavior':<20} {'True Count':<15} {'Predicted Count':<15}")
    print("="*50)

    for behavior in behaviors_train:
        true_count = count_behavior_instances(y_train, behavior)
        pred_count = count_behavior_instances(y_train_pred, behavior)
        print(f"{behavior:<20} {true_count:<15} {pred_count:<15}")

    print("\n=== Behavior Instance Counts - TEST SET ===")
    behaviors = np.unique(y_test)

    print(f"{'Behavior':<20} {'True Count':<15} {'Predicted Count':<15}")
    print("="*50)

    # Collect counts for plotting
    true_counts = {}
    pred_counts = {}
    for behavior in behaviors:
        true_count = count_behavior_instances(y_test, behavior)
        pred_count = count_behavior_instances(y_pred, behavior)
        true_counts[behavior] = true_count
        pred_counts[behavior] = pred_count
        print(f"{behavior:<20} {true_count:<15} {pred_count:<15}")

    # Plot behavior instance counts for specific behaviors
    plot_behaviors = ['supportedrear', 'unsupportedrear', 'grooming']
    available_behaviors = [b for b in plot_behaviors if b in behaviors]

    if available_behaviors:
        fig, axes = plt.subplots(1, len(available_behaviors), figsize=(5*len(available_behaviors), 5))
        if len(available_behaviors) == 1:
            axes = [axes]

        colors_true = '#59a89c'
        colors_pred = '#a559aa'

        for idx, behavior in enumerate(available_behaviors):
            true_val = true_counts[behavior]
            pred_val = pred_counts[behavior]

            x_pos = np.array([0, 1])
            values = [true_val, pred_val]
            bars = axes[idx].bar(x_pos, values, color=[colors_true, colors_pred], width=0.6)

            # Add count values above bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(height)}',
                             ha='center', va='bottom', fontsize=13, fontweight='bold')

            axes[idx].set_title(f'{behavior}', fontsize=15, fontweight='bold', pad=10)
            axes[idx].set_ylabel('Instance Count', fontsize=14, labelpad=10)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(['True', 'Predicted'], fontsize=14)
            axes[idx].tick_params(axis='y', labelsize=14)
            axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
            axes[idx].set_ylim([0, max(values) * 1.15])

        fig.suptitle('Behaviour Instance Counts (True vs Predicted)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('pipeline_outputs/behaviour_instance_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nBehaviour instance count plot saved!")

    ################################### SCATTERPLOT: ACTUAL vs PREDICTED INSTANCES PER VIDEO ###############################################################

    # Create scatterplot comparing actual vs predicted instance counts per video
    # Check if video_id exists in either columns or index
    has_video_id = ('video_id' in X_test.columns) or ('video_id' in X_test.index.names)

    if has_video_id:
        # Get video_ids from index or columns
        if 'video_id' in X_test.index.names:
            video_ids = X_test.index.get_level_values('video_id').values
        else:
            video_ids = X_test['video_id'].values

        unique_videos = np.unique(video_ids)

        # Collect instance counts per video for each behavior
        video_counts = {behavior: {'actual': [], 'predicted': []} for behavior in available_behaviors}

        for video in unique_videos:
            video_mask = (video_ids == video)
            y_test_video = y_test[video_mask]
            y_pred_video = y_pred[video_mask]

            for behavior in available_behaviors:
                actual_count = count_behavior_instances(y_test_video, behavior)
                predicted_count = count_behavior_instances(y_pred_video, behavior)
                video_counts[behavior]['actual'].append(actual_count)
                video_counts[behavior]['predicted'].append(predicted_count)

        # Create scatterplot with subplots for each behavior
        fig, axes = plt.subplots(1, 3 , figsize=(15, 5))

        for idx, behavior in enumerate(available_behaviors):
            actual = np.array(video_counts[behavior]['actual'])
            predicted = np.array(video_counts[behavior]['predicted'])

            # Scatterplot
            axes[idx].scatter(actual, predicted, alpha=0.6, s=100)

            # Add diagonal line (perfect prediction)
            max_val = max(actual.max(), predicted.max()) if len(actual) > 0 else 1
            axes[idx].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect Agreement')

            axes[idx].set_xlabel('Actual Instance Count', fontsize=14)
            axes[idx].set_ylabel('Predicted Instance Count', fontsize=14)
            axes[idx].set_title(f'{behavior}', fontsize=17, fontweight='bold', pad=10)
            axes[idx].legend(fontsize=10.5)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal', adjustable='box')

        fig.suptitle('Actual vs Predicted Instance Counts Per Video', fontsize=20, fontweight='bold', y=1)
        plt.tight_layout()
        plt.savefig('pipeline_outputs/instance_count_per_video_scatterplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nInstance count per video scatterplot saved!")




