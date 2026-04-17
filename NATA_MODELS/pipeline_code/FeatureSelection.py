import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np



def permutation_importance(model, X_train, X_test, y_train, y_test, n_repeats=10, random_state=42, n_jobs=2):

    # Extract feature importances using Permutation Importance
    print("Calculating permutation importance...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats,
        random_state,
        n_jobs
    )
    importances = result.importances_mean
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Filter features with importance > 0
    feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
    print(f"Features with importance > 0: {len(feature_importance_df)}")
    print(feature_importance_df.head(20))

    # Plot top 20 feature importances
    top_n_plot = 20
    top_features_plot = feature_importance_df.head(top_n_plot)
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(top_features_plot)), top_features_plot['Importance'], align='center')
    plt.yticks(range(len(top_features_plot)), top_features_plot['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    model_name = "Histogram Gradient Boosting"
    plt.title(f'Top {top_n_plot} {model_name} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('pipeline_outputs/feature_importances_HGB.png', dpi=300, bbox_inches='tight')
    plt.close()

    return feature_importance_df

def feature_importance(model, X_train):

    # Extract feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df.head(10))


    # Plot top 20 feature importances
    top_n_plot = 20
    top_features_plot = feature_importance_df.head(top_n_plot)
    plt.figure(figsize=(10, 12))
    plt.barh(range(top_n_plot), top_features_plot['Importance'], align='center')
    plt.yticks(range(top_n_plot), top_features_plot['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n_plot} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('pipeline_outputs/RF_feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

    return feature_importance_df