import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



_DISPLAY_NAMES = {
    'supportedrearing':   'Supported Rearing',
    'unsupportedrearing': 'Unsupported Rearing',
}

def _fmt(name):
    return _DISPLAY_NAMES.get(name.lower(), name.capitalize())


# Count Comparison Scatterplot

def plot_instance_count_scatter(model_wrappers, output_path):
    """
    Scatterplot of true vs predicted behavior instance counts per video,
    overlaying multiple models in different colors.

    Args:
        model_wrappers: list of ModelWrapper objects
        output_path:    path to save the PNG
    """
    column_names    = model_wrappers[0].column_names
    behavior_indices = [idx for idx in sorted(column_names.keys()) if idx != 0]
    behavior_labels  = [column_names[idx] for idx in behavior_indices]

    n    = len(behavior_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Paul Tol "bright" colorblind-friendly palette
    model_colors  = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
    model_markers = ['o', 's', '^', 'D', 'P', 'X']

    palette  = {m.name: c for m, c in zip(model_wrappers, model_colors)}
    markers  = {m.name: mk for m, mk in zip(model_wrappers, model_markers)}

    # Build tidy DataFrame
    rows = []
    for model in model_wrappers:
        for beh_idx in behavior_indices:
            for true_val, pred_val in zip(model.true_behavior_count, model.pred_behavior_count):
                rows.append({
                    'model':     model.name,
                    'behavior':  column_names[beh_idx],
                    'predicted': pred_val[beh_idx],
                    'true':      true_val[beh_idx],
                })
    df = pd.DataFrame(rows)

    for ax, beh_idx, beh_name in zip(axes, behavior_indices, behavior_labels):
        beh_df = df[df['behavior'] == beh_name]

        sns.scatterplot(data=beh_df, x='predicted', y='true',
                        hue='model', style='model',
                        palette=palette, markers=markers,
                        alpha=0.85, s=90, ax=ax, zorder=3)

        max_val = max(beh_df[['predicted', 'true']].max()) if not beh_df.empty else 1

        lim = max_val * 1.1 if max_val > 0 else 1

        # Regression per model (with intercept)
        for model, color in zip(model_wrappers, model_colors):
            sub = beh_df[beh_df['model'] == model.name]
            if len(sub) >= 2:
                x = sub['predicted'].values.reshape(-1, 1)

                # add intercept column
                X = np.hstack([x, np.ones_like(x)])

                slope, intercept = np.linalg.lstsq(X, sub['true'].values, rcond=None)[0]

                x_range = np.linspace(0, lim, 100)
                ax.plot(x_range, slope * x_range + intercept,
                        color=color, linewidth=1.5, alpha=0.6)

        ax.plot([0, lim], [0, lim], 'k--', alpha=0.35, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(_fmt(beh_name), fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Count', fontsize=12)
        ax.set_ylabel('True Count', fontsize=12)
        ax.get_legend().remove()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle('Behaviour Instance Counts — True vs Predicted', fontsize=15, fontweight='bold', y=1.05)

    # Single shared legend below title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(model_wrappers),
               fontsize=10, bbox_to_anchor=(0.5, 1.00), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# F1 Scores

def plot_f1_scores(model_wrappers, output_path):
    """
    Grouped bar chart of per-class F1 scores across multiple models.

    Args:
        model_wrappers: list of ModelWrapper objects (each must have .f1_scores dict)
        output_path:    path to save the PNG
    """
    # Paul Tol "bright" palette — same order as scatter plot
    model_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']

    # Behaviour order: background first, then the rest sorted
    column_names = model_wrappers[0].column_names
    behavior_names = [column_names[k] for k in sorted(column_names.keys())]

    n_behaviors = len(behavior_names)
    n_models    = len(model_wrappers)
    width       = 0.8 / n_models
    x_pos       = np.arange(n_behaviors)

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_behaviors), 6))

    for i, (model, color) in enumerate(zip(model_wrappers, model_colors)):
        scores = [model.f1_scores.get(b, 0.0) for b in behavior_names]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x_pos + offset, scores, width, label=model.name, color=color)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('F1 Score per Behaviour Class', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('F1 Score', fontsize=14, labelpad=12)
    ax.set_xlabel('Behaviour Class', fontsize=14, labelpad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_fmt(b) for b in behavior_names], rotation=0, ha='center', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=12, ncol=min(n_models, 4), loc='upper right', columnspacing=1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Computing Times

def plot_computing_times(output_path):
    """
    Bar chart comparing average inference time per frame across models.
    Fill in the times_ms dict before running.

    Args:
        output_path: path to save the PNG
    """

    times_s = {
        'Old Model': 89, #89s
        'HGB': 0, #feature generation (524.9s /54 * 10) + 13.4s
        'Transformer': 1000, #feature generation 524.9s + 18.5s
        'TCNN': 0,
        'CNN Transformer': 0, # rotate vidoes + 248.6s (4.1min)
    }

    model_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']

    names  = list(times_s.keys())
    values = list(times_s.values())
    colors = model_colors[:len(names)]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(names)), 5))

    bars = ax.bar(names, values, color=colors, width=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f} s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Computing Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Time (s)', fontsize=14, labelpad=12)
    ax.set_xlabel('Model', fontsize=14, labelpad=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
