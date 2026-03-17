import numpy as np
import matplotlib.pyplot as plt

# Hardcoded F1 scores for 4 models (test scores only)
F_model_f1 = {
    'background': 0.91,
    'grooming': 0.83,
    'supportedrear': 0.87,
    'unsupportedrear': 0.37,
    'digging': 0.75
}

HGB_f1 = {
    'background': 0.80,
    'grooming': 0.66,
    'supportedrear': 0.68,
    'unsupportedrear': 0.23,
    'digging': 0.70
}

#
LSTM_f1 = {
    'background': 0.88,
    'grooming': 0.75,
    'supportedrear': 0.76,
    'unsupportedrear': 0.30,
    'digging': 0.77
}

Transformer_f1 = {
    'background': 0.89,
    'grooming': 0.77,
    'supportedrear': 0.81,
    'unsupportedrear': 0.39,
    'digging': 0.79
}

# Create comparison bar plot
behaviors = ['background', 'supportedrear', 'unsupportedrear', 'grooming', 'digging']
model1_scores = [F_model_f1[b] for b in behaviors]
model2_scores = [HGB_f1[b] for b in behaviors]
model3_scores = [LSTM_f1[b] for b in behaviors]
model4_scores = [Transformer_f1[b] for b in behaviors]

x_pos = np.arange(len(behaviors))
width = 0.2

plt.figure(figsize=(14, 6))
bars1 = plt.bar(x_pos - 1.5*width, model1_scores, width, label='Current Model', color='#4477AA')
bars2 = plt.bar(x_pos - 0.5*width, model2_scores, width, label='HGB', color='#66CCEE')
bars3 = plt.bar(x_pos + 0.5*width, model3_scores, width, label='LSTM', color='#228833')
bars4 = plt.bar(x_pos + 1.5*width, model4_scores, width, label='Transformer', color='#EE6677')

# Add F1 score values above bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars4:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('F1 Score per Behavior Class', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('F1 Score', fontsize=16, labelpad=15)
plt.xlabel('Behaviour Class', fontsize=16, labelpad=15)
plt.xticks(x_pos, behaviors, rotation=0, ha='center', fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0, 1.05])
plt.legend(fontsize=13, ncol=2, loc='upper right', columnspacing=1.0)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('pipeline_outputs/f1_scores_model_comparison_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("F1 score model comparison bar plot saved!")
