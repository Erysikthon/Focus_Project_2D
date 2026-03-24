import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ==================================================
# Settings
# ==================================================
ROOT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = ROOT_DIR / "_artifacts" / "features"
LABELS_DIR = ROOT_DIR / "labels"

TEST_SIZE = 0.2
RANDOM_STATE = 42

BEHAVIOR_COLUMNS = ["background", "supportedrear", "unsupportedrear", "grooming"]

# Downsample only in training
BACKGROUND_LABEL = "background"
BACKGROUND_TO_MAX_RATIO = 2.0   # e.g. keep at most 2x as many background rows as all non-background rows

# Optional temporal smoothing of numeric features within each recording
USE_TEMPORAL_SMOOTHING = True
SMOOTHING_WINDOW = 5   # rolling mean window in frames


# ==================================================
# Helper
# ==================================================
def print_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def sort_mixed(values):
    return sorted(values, key=lambda x: int(x) if str(x).isdigit() else str(x))


def downsample_background(df, background_label="background", max_ratio=2.0, random_state=42):
    """
    Keep all non-background rows.
    Keep at most max_ratio * n_non_background background rows.
    """
    bg = df[df["label"] == background_label]
    non_bg = df[df["label"] != background_label]

    if len(non_bg) == 0:
        return df.copy()

    max_bg = int(max_ratio * len(non_bg))

    if len(bg) <= max_bg:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    bg_sampled = bg.sample(n=max_bg, random_state=random_state)
    out = pd.concat([bg_sampled, non_bg], axis=0)
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out


# ==================================================
# LOAD FEATURES AND LABELS
# ==================================================
print_section("LOAD FEATURES AND LABELS")

feature_files = sorted((FEATURES_DIR / "elements").glob("*/data.csv"))
label_files = sorted(LABELS_DIR.glob("*.csv"))

print("ROOT_DIR:", ROOT_DIR)
print("FEATURES_DIR:", FEATURES_DIR)
print("LABELS_DIR:", LABELS_DIR)

print("\nFeature files found:")
for f in feature_files:
    print("  ", f)

print("\nLabel files found:")
for f in label_files:
    print("  ", f.name)

dfs = []

for feature_path in feature_files:
    handle = feature_path.parent.name
    label_path = LABELS_DIR / f"{handle}.csv"

    print("\n" + "-" * 60)
    print("HANDLE:", handle)

    if not label_path.exists():
        print("-> skipped: no matching label file")
        continue

    df_features = pd.read_csv(feature_path)
    df_labels = pd.read_csv(label_path)

    print("Feature columns:", list(df_features.columns[:15]))
    print("Label columns:", list(df_labels.columns))

    # -----------------------------
    # Normalize frame column names
    # -----------------------------
    if "frame" in df_features.columns and "frame_index" not in df_features.columns:
        df_features = df_features.rename(columns={"frame": "frame_index"})

    if "Unnamed: 0" in df_labels.columns and "frame_index" not in df_labels.columns:
        df_labels = df_labels.rename(columns={"Unnamed: 0": "frame_index"})

    if "frame_index" not in df_features.columns:
        print("-> skipped: frame_index missing in feature file")
        continue

    if "frame_index" not in df_labels.columns:
        print("-> skipped: frame_index missing in label file")
        continue

    # -----------------------------
    # Check label columns
    # -----------------------------
    missing_behavior_cols = [c for c in BEHAVIOR_COLUMNS if c not in df_labels.columns]
    if missing_behavior_cols:
        print(f"-> skipped: missing behavior columns {missing_behavior_cols}")
        continue

    # -----------------------------
    # Convert one-hot labels to one label column
    # Keep only rows with exactly one active class
    # -----------------------------
    label_sums = df_labels[BEHAVIOR_COLUMNS].sum(axis=1)
    valid_label_rows = label_sums == 1

    print("Rows in label file:", len(df_labels))
    print("Rows with exactly one active label:", int(valid_label_rows.sum()))

    df_labels = df_labels.loc[valid_label_rows, ["frame_index"] + BEHAVIOR_COLUMNS].copy()
    df_labels["label"] = df_labels[BEHAVIOR_COLUMNS].idxmax(axis=1)

    # -----------------------------
    # Merge features and labels
    # -----------------------------
    df = pd.merge(
        df_features,
        df_labels[["frame_index", "label"]],
        on="frame_index",
        how="inner"
    )

    print("Merged rows:", len(df))

    if len(df) == 0:
        print("-> skipped: no overlapping frames")
        continue

    # -----------------------------
    # Remove non-feature columns
    # -----------------------------
    X = df.drop(columns=["frame_index", "label"]).copy()
    y = df["label"].copy()

    X = X.select_dtypes(include=["number", "bool"]).copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    valid_feature_rows = ~X.isna().any(axis=1)

    print("Rows before NaN filter:", len(X))
    print("Rows after NaN filter:", int(valid_feature_rows.sum()))

    X = X.loc[valid_feature_rows].copy()
    y = y.loc[valid_feature_rows].copy()
    frame_index = df.loc[valid_feature_rows, "frame_index"].copy()

    if len(X) == 0:
        print("-> skipped: no valid rows after removing NaNs")
        continue

    df_clean = X.copy()
    df_clean["frame_index"] = frame_index.values
    df_clean["label"] = y.values
    df_clean["group"] = handle

    # -----------------------------
    # Optional temporal smoothing
    # numeric feature columns only, within recording
    # -----------------------------
    if USE_TEMPORAL_SMOOTHING:
        feature_cols = [c for c in df_clean.columns if c not in {"frame_index", "label", "group"}]
        df_clean = df_clean.sort_values("frame_index").reset_index(drop=True)
        df_clean[feature_cols] = (
            df_clean[feature_cols]
            .rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True)
            .mean()
        )

    dfs.append(df_clean)
    print("-> accepted")

if len(dfs) == 0:
    raise ValueError("No usable feature/label pairs found.")

df_all = pd.concat(dfs, axis=0, ignore_index=True)

print_section("DATASET SUMMARY")
print("Total samples:", len(df_all))
print("Total features:", len([c for c in df_all.columns if c not in {"frame_index", "label", "group"}]))
print("\nClass counts:")
print(df_all["label"].value_counts())
print("\nSamples per recording:")
print(df_all["group"].value_counts().sort_index())


# ==================================================
# GROUP-BASED TRAIN / TEST SPLIT
# ==================================================
print_section("GROUP-BASED TRAIN / TEST SPLIT")

X_all = df_all.drop(columns=["label", "group"])
y_all = df_all["label"]
groups = df_all["group"]

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))

X_train = X_all.iloc[train_idx].copy()
X_test = X_all.iloc[test_idx].copy()
y_train = y_all.iloc[train_idx].copy()
y_test = y_all.iloc[test_idx].copy()
groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

print("Train recordings:", sort_mixed(groups_train.unique()))
print("Test recordings:", sort_mixed(groups_test.unique()))
print(f"Train samples before downsampling: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTrain class counts before downsampling:")
print(y_train.value_counts())

print("\nTest class counts:")
print(y_test.value_counts())


# ==================================================
# TRAINING DOWNSAMPLING
# ==================================================
print_section("TRAINING DOWNSAMPLING")

train_df = X_train.copy()
train_df["label"] = y_train.values

train_df_balanced = downsample_background(
    train_df,
    background_label=BACKGROUND_LABEL,
    max_ratio=BACKGROUND_TO_MAX_RATIO,
    random_state=RANDOM_STATE,
)

X_train_bal = train_df_balanced.drop(columns=["label"]).copy()
y_train_bal = train_df_balanced["label"].copy()

print(f"Train samples after downsampling: {len(X_train_bal)}")
print("\nTrain class counts after downsampling:")
print(y_train_bal.value_counts())


# ==================================================
# TRAIN RANDOM FOREST
# ==================================================
print_section("TRAIN RANDOM FOREST")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=5,
)

model.fit(X_train_bal, y_train_bal)


# ==================================================
# EVALUATE
# ==================================================
print_section("EVALUATE")

y_pred = model.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred, digits=3))

labels_sorted = sorted(y_all.unique())

print("\nConfusion matrix (counts):")
cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=labels_sorted),
    index=labels_sorted,
    columns=labels_sorted,
)
print(cm)

print("\nConfusion matrix (row-normalized):")
cm_norm = cm.div(cm.sum(axis=1), axis=0).round(3)
print(cm_norm)

accuracy = (y_pred == y_test).mean()
print(f"\nOverall accuracy: {accuracy:.3f}")


# ==================================================
# PER-RECORDING PERFORMANCE
# ==================================================
print_section("PER-RECORDING PERFORMANCE")

test_results = pd.DataFrame({
    "group": groups_test.values,
    "true": y_test.values,
    "pred": y_pred,
})

for handle in sort_mixed(test_results["group"].unique()):
    df_rec = test_results[test_results["group"] == handle]
    rec_acc = (df_rec["true"] == df_rec["pred"]).mean()
    print(f"Recording {handle}: n={len(df_rec)}, accuracy={rec_acc:.3f}")


# ==================================================
# TOP FEATURE IMPORTANCE
# ==================================================
print_section("TOP FEATURE IMPORTANCE")

importances = pd.Series(model.feature_importances_, index=X_train_bal.columns)
importances = importances.sort_values(ascending=False)

print(importances.head(20))