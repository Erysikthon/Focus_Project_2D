import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Pie chart of class imbalance
y_path = "./pipeline_saved_processes/dataframes/y.csv"
y = pd.read_csv(y_path, index_col=["video_id", "frame"])
y_flat = y.values.ravel()
unique_classes, class_counts = np.unique(y_flat, return_counts=True)

# Colorblind-friendly colors (Tol palette - less orange)
colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    class_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors[:len(unique_classes)],
    textprops={'fontsize': 15, 'weight': 'bold'},
    radius=0.7,
    pctdistance=0.75
)

# Make percentage text more legible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(17)
    autotext.set_weight('bold')

# Add legend
ax.legend(wedges, unique_classes, title="Behaviours", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=16, title_fontsize=18)

plt.title('Class Distribution in Dataset', fontsize=19, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig('pipeline_outputs/class_imbalance_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")

# F1 Score Table - Hardcoded metrics
print("Creating F1 score comparison table...")

# Hardcoded F1 scores
train_f1 = {
    'background': 0.99,
    'grooming': 0.88,
    'supportedrear': 0.96,
    'unsupportedrear': 0.94
}

test_f1 = {
    'background': 0.98,
    'grooming': 0.75,
    'supportedrear': 0.89,
    'unsupportedrear': 0.73
}

# Create comparison bar plot
behaviors = ['background', 'supportedrear', 'unsupportedrear', 'grooming']
train_scores = [train_f1[b] for b in behaviors]
test_scores = [test_f1[b] for b in behaviors]

x_pos = np.arange(len(behaviors))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x_pos - width/2, train_scores, width, label='Train', color='#A8DADC')
bars2 = plt.bar(x_pos + width/2, test_scores, width, label='Test', color='#E8C5A0')

# Add F1 score values above bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=13, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.title('F1 Score per Behavior Class (Train vs Test)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('F1 Score', fontsize=16, labelpad=15)
plt.xlabel('Behaviour Class', fontsize=16, labelpad=15)
plt.xticks(x_pos, behaviors, rotation=0, ha='center', fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0, 1.05])
plt.legend(fontsize=13)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('pipeline_outputs/f1_scores_model_1_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("F1 score bar plot saved!")

# Feature Importance Graph
feature_importance_path = './pipeline_saved_processes/selected_features/HGB_final_selected_features.csv'

if os.path.isfile(feature_importance_path):
 print("Loading existing permutation importance...")
 feature_importance_df = pd.read_csv(feature_importance_path)
 print(f"Features with importance > 0: {len(feature_importance_df)}")
 print(feature_importance_df.head(20))

 # Plot top 100 feature importances
 top_n_plot = 100
 top_features_plot = feature_importance_df.head(top_n_plot)
 plt.figure(figsize=(15, 25))
 plt.barh(range(len(top_features_plot)), top_features_plot['Importance'], align='center')
 plt.yticks(range(len(top_features_plot)), top_features_plot['Feature'])
 plt.xlabel('Importance', fontsize=20, labelpad=30)
 plt.ylabel('Feature', fontsize=20, labelpad=30)
 model_name = "Histogram Gradient Boosting"
 plt.title(f'Top {top_n_plot} Permutation Importances', fontsize=30, pad=42, fontweight='bold')
 plt.gca().invert_yaxis()
 plt.subplots_adjust(left=0.4)
 plt.savefig('pipeline_outputs/feature_importances_HGB_final.png', dpi=300, bbox_inches='tight')
 plt.close()



